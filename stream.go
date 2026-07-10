package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// StreamProcessor menangani pemrosesan RTSP stream menggunakan FFmpeg
type StreamProcessor struct {
	rtspURL       string
	aiClient      *AIClient
	db            *DBManager
	confThreshold float64

	// Koordinat zona memasak (ROI) dalam persentase (0.0 - 1.0)
	zoneXMin float64
	zoneYMin float64
	zoneXMax float64
	zoneYMax float64

	// Logika pendeteksi aktivitas memasak
	cookingTriggerSecs int
	inZoneStartTime    time.Time
	isCooking          bool
	lastSeenInZone     time.Time
	kitchenStatus      string // "Kosong", "Memasak"
	secondsInZone      int
	
	clients       map[chan []byte]bool
	clientsMu     sync.Mutex
	
	currentFrame  []byte
	currentFrameMu sync.RWMutex
	
	running       bool
	runningMu     sync.Mutex
	stopChan      chan struct{}
	
	fps           float64
	aiLatency     time.Duration
	aiStatus      string // "Online", "Offline"
}

// NewStreamProcessor membuat instansi baru StreamProcessor
func NewStreamProcessor(
	rtspURL string, 
	aiClient *AIClient, 
	db *DBManager, 
	confThreshold float64,
	zxMin, zyMin, zxMax, zyMax float64,
	cookingTriggerSecs int,
) *StreamProcessor {
	return &StreamProcessor{
		rtspURL:            rtspURL,
		aiClient:           aiClient,
		db:                 db,
		confThreshold:      confThreshold,
		zoneXMin:           zxMin,
		zoneYMin:           zyMin,
		zoneXMax:           zxMax,
		zoneYMax:           zyMax,
		cookingTriggerSecs: cookingTriggerSecs,
		kitchenStatus:      "Kosong",
		clients:            make(map[chan []byte]bool),
		stopChan:           make(chan struct{}),
		aiStatus:           "Unknown",
	}
}

// getFFmpegPath mencari path instalasi ffmpeg
func getFFmpegPath() string {
	// Coba cari di environment PATH
	if path, err := exec.LookPath("ffmpeg"); err == nil {
		return path
	}

	// Cari di folder local appdata winget (biasanya untuk Gyan.FFmpeg)
	localAppData := os.Getenv("LOCALAPPDATA")
	if localAppData != "" {
		wingetDir := filepath.Join(localAppData, "Microsoft", "WinGet", "Packages")
		matches, err := filepath.Glob(filepath.Join(wingetDir, "*FFmpeg*", "*", "bin", "ffmpeg.exe"))
		if err == nil && len(matches) > 0 {
			return matches[0]
		}
		
		matches, err = filepath.Glob(filepath.Join(wingetDir, "*FFmpeg*", "*", "ffmpeg.exe"))
		if err == nil && len(matches) > 0 {
			return matches[0]
		}

		matches, err = filepath.Glob(filepath.Join(wingetDir, "*FFmpeg*", "bin", "ffmpeg.exe"))
		if err == nil && len(matches) > 0 {
			return matches[0]
		}
	}

	// Default gunakan perintah "ffmpeg" dengan harapan terinstal di sistem
	return "ffmpeg"
}

// splitJPEG memisahkan data input berdasarkan marker SOI (0xFFD8) dan EOI (0xFFD9) JPEG
func splitJPEG(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	
	// Cari SOI (Start Of Image)
	soi := bytes.Index(data, []byte{0xFF, 0xD8})
	if soi == -1 {
		return 0, nil, nil
	}
	
	// Cari EOI (End Of Image) dari titik SOI
	eoi := bytes.Index(data[soi:], []byte{0xFF, 0xD9})
	if eoi == -1 {
		return 0, nil, nil
	}
	
	endIdx := soi + eoi + 2
	return endIdx, data[soi:endIdx], nil
}

// Start menjalankan loop pemrosesan video
func (sp *StreamProcessor) Start() {
	sp.runningMu.Lock()
	if sp.running {
		sp.runningMu.Unlock()
		return
	}
	sp.running = true
	sp.stopChan = make(chan struct{})
	sp.runningMu.Unlock()

	ffmpegPath := getFFmpegPath()
	log.Printf("Menggunakan FFmpeg path: %s", ffmpegPath)

	go func() {
		for {
			select {
			case <-sp.stopChan:
				return
			default:
				log.Println("Memulai koneksi ke RTSP stream via FFmpeg...")
				
				// Argumen ffmpeg untuk mengambil stream RTSP dan menyajikan gambar JPEG di stdout
				cmd := exec.Command(ffmpegPath,
					"-rtsp_transport", "tcp",
					"-i", sp.rtspURL,
					"-f", "image2pipe",
					"-vcodec", "mjpeg",
					"-q:v", "4", // Kualitas gambar
					"-pix_fmt", "yuvj420p",
					"-",
				)

				stdout, err := cmd.StdoutPipe()
				if err != nil {
					log.Printf("Gagal membuat stdout pipe FFmpeg: %v. Mencoba kembali dalam 3 detik...", err)
					time.Sleep(3 * time.Second)
					continue
				}

				if err := cmd.Start(); err != nil {
					log.Printf("Gagal menjalankan FFmpeg: %v. Pastikan FFmpeg terinstal. Mencoba kembali dalam 3 detik...", err)
					time.Sleep(3 * time.Second)
					continue
				}

				// Scanner dengan buffer besar (maks 2MB)
				scanner := bufio.NewScanner(stdout)
				buf := make([]byte, 1024*1024)
				scanner.Buffer(buf, 2*1024*1024)
				scanner.Split(splitJPEG)

				lastAIDetectTime := time.Time{}
				frameCount := 0
				fpsLastTime := time.Now()

				for scanner.Scan() {
					select {
					case <-sp.stopChan:
						cmd.Process.Kill()
						return
					default:
					}

					jpegData := scanner.Bytes()
					if len(jpegData) == 0 {
						continue
					}

					frameCount++
					if frameCount >= 30 {
						now := time.Now()
						elapsed := now.Sub(fpsLastTime).Seconds()
						sp.fps = float64(frameCount) / elapsed
						frameCount = 0
						fpsLastTime = now
					}

					// Batasi pengiriman ke AI (sekitar 5 frame per detik / setiap 200ms)
					var detections []AIDetection
					var aiErr error
					now := time.Now()
					
					runAI := now.Sub(lastAIDetectTime) >= 200*time.Millisecond

					if runAI {
						lastAIDetectTime = now
						aiStart := time.Now()
						detections, aiErr = sp.aiClient.Detect(jpegData)
						sp.aiLatency = time.Since(aiStart)
						
						if aiErr != nil {
							sp.aiStatus = "Offline"
						} else {
							sp.aiStatus = "Online"
						}
					}

					// Proses logika deteksi ROI dapur & penggambaran bounding box
					var processedJPEG []byte
					if aiErr == nil {
						processedJPEG, _, _ = sp.drawBoundingBoxes(jpegData, detections)
					} else {
						// Fallback ketika AI Offline: tetap update timer zona (agar tidak nge-lock)
						// dan tetap gambar garis zona di layar video
						processedJPEG, _, _ = sp.drawBoundingBoxes(jpegData, nil)
						
						if runAI {
							log.Printf("Layanan AI offline: %v", aiErr)
						}
					}

					// Simpan frame terbaru untuk klien baru
					sp.currentFrameMu.Lock()
					sp.currentFrame = processedJPEG
					sp.currentFrameMu.Unlock()

					// Siarkan ke seluruh klien browser yang terhubung
					sp.broadcast(processedJPEG)
				}

				if err := scanner.Err(); err != nil {
					log.Printf("Error pembacaan stream scanner: %v", err)
				}

				cmd.Process.Kill()
				cmd.Wait()
				log.Println("Proses FFmpeg berhenti. Melakukan koneksi ulang dalam 3 detik...")
				time.Sleep(3 * time.Second)
			}
		}
	}()
}

// Stop menghentikan pemrosesan stream
func (sp *StreamProcessor) Stop() {
	sp.runningMu.Lock()
	if !sp.running {
		sp.runningMu.Unlock()
		return
	}
	sp.running = false
	close(sp.stopChan)
	sp.runningMu.Unlock()
}

// AddClient mendaftarkan klien baru untuk menerima frame MJPEG
func (sp *StreamProcessor) AddClient(ch chan []byte) {
	sp.clientsMu.Lock()
	sp.clients[ch] = true
	sp.clientsMu.Unlock()
}

// RemoveClient menghapus klien dari daftar siaran
func (sp *StreamProcessor) RemoveClient(ch chan []byte) {
	sp.clientsMu.Lock()
	delete(sp.clients, ch)
	sp.clientsMu.Unlock()
}

// broadcast mengirimkan frame ke seluruh klien terdaftar
func (sp *StreamProcessor) broadcast(frame []byte) {
	sp.clientsMu.Lock()
	defer sp.clientsMu.Unlock()
	for ch := range sp.clients {
		select {
		case ch <- frame:
		default:
			// Lewati klien jika channel penuh untuk mencegah lag sistem
		}
	}
}

// drawBoundingBoxes menggambar kotak pembatas objek dan zona memasak pada frame JPEG
func (sp *StreamProcessor) drawBoundingBoxes(jpegData []byte, detections []AIDetection) ([]byte, bool, float64) {
	// Decode gambar JPEG ke image.Image
	srcImg, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		return jpegData, false, 0
	}

	bounds := srcImg.Bounds()
	width := float64(bounds.Max.X - bounds.Min.X)
	height := float64(bounds.Max.Y - bounds.Min.Y)

	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, bounds, srcImg, bounds.Min, draw.Src)

	personInZone := false
	maxConf := 0.0

	// Definisikan warna-warna box
	colorRed := color.RGBA{R: 239, G: 68, B: 68, A: 255}   // Merah cerah
	colorYellow := color.RGBA{R: 245, G: 158, B: 11, A: 255} // Kuning
	colorBlue := color.RGBA{R: 59, G: 130, B: 246, A: 255}   // Biru

	// Proses setiap objek terdeteksi
	for _, det := range detections {
		isPerson := det.Label == "person"
		
		if det.Confidence < sp.confThreshold {
			continue
		}

		// Koordinat box absolut
		x1, y1 := int(det.Box[0]), int(det.Box[1])
		x2, y2 := int(det.Box[2]), int(det.Box[3])

		// Batasi koordinat ke dalam layar gambar
		if x1 < 0 { x1 = 0 }
		if y1 < 0 { y1 = 0 }
		if x2 >= bounds.Max.X { x2 = bounds.Max.X - 1 }
		if y2 >= bounds.Max.Y { y2 = bounds.Max.Y - 1 }

		// Pilih warna berdasarkan jenis objek
		var boxColor color.Color = colorYellow
		if isPerson {
			boxColor = colorRed

			// LOGIKA DETEKSI ZONA MEMASAK (ROI):
			// Kita ambil posisi kaki orang tersebut (tengah bawah dari bounding box):
			// px = (x1 + x2) / 2
			// py = y2
			px := (x1 + x2) / 2
			py := y2

			// Normalisasikan koordinat absolut menjadi koordinat rasio persen (0.0 - 1.0)
			normX := float64(px) / width
			normY := float64(py) / height

			// Cek apakah koordinat kaki masuk ke dalam area Zona ROI
			if normX >= sp.zoneXMin && normX <= sp.zoneXMax && normY >= sp.zoneYMin && normY <= sp.zoneYMax {
				personInZone = true
				if det.Confidence > maxConf {
					maxConf = det.Confidence
				}
			}
		}

		// Gambar kotak pembatas objek
		drawBox(rgbaImg, x1, y1, x2, y2, boxColor, 3)
	}

	// STATE MACHINE DETEKSI MEMASAK
	now := time.Now()
	if personInZone {
		if sp.inZoneStartTime.IsZero() {
			sp.inZoneStartTime = now
		}
		sp.lastSeenInZone = now
		
		// Hitung durasi saat ini berada di dalam zona kompor
		elapsed := now.Sub(sp.inZoneStartTime)
		sp.secondsInZone = int(elapsed.Seconds())
		
		// Jika berada dalam zona melebihi batas waktu kompor, pemicu status memasak aktif
		if sp.secondsInZone >= sp.cookingTriggerSecs {
			if !sp.isCooking {
				sp.isCooking = true
				sp.kitchenStatus = "Memasak"
				
				// PENTING: Jalankan penyimpanan snapshot & log ke database secara asinkron
				go sp.saveCookingSnapshot(jpegData, maxConf)
			}
		}
	} else {
		// Jika tidak ada orang di dalam zona
		if !sp.inZoneStartTime.IsZero() {
			// Berikan toleransi waktu 3 detik (grace period) jika deteksi terputus sesaat
			if now.Sub(sp.lastSeenInZone) >= 3*time.Second {
				sp.inZoneStartTime = time.Time{}
				sp.isCooking = false
				sp.kitchenStatus = "Kosong"
				sp.secondsInZone = 0
			}
		} else {
			sp.kitchenStatus = "Kosong"
			sp.secondsInZone = 0
		}
	}

	// GAMBAR KOTAK ZONA MEMASAK (ROI)
	// Ubah koordinat rasio ROI kembali menjadi koordinat absolut piksel gambar
	zx1 := int(sp.zoneXMin * width)
	zy1 := int(sp.zoneYMin * height)
	zx2 := int(sp.zoneXMax * width)
	zy2 := int(sp.zoneYMax * height)

	// Warna kotak zona:
	// - Merah: Jika status memasak aktif terdeteksi
	// - Kuning: Jika ada orang masuk zona, tapi belum memicu durasi threshold
	// - Biru: Jika zona kosong
	var zoneColor color.Color = colorBlue
	if sp.isCooking {
		zoneColor = colorRed
	} else if personInZone {
		zoneColor = colorYellow
	}

	// Gambar kotak zona tipis
	drawBox(rgbaImg, zx1, zy1, zx2, zy2, zoneColor, 2)
	
	// Gambar label tag "ZONA MEMASAK" di atas kotak zona
	drawFilledRect(rgbaImg, zx1, zy1, zx1+100, zy1+18, zoneColor)

	// GAMBAR BANNER STATUS DAPUR DI KIRI ATAS FRAME
	// Banner akan memvisualisasikan status saat ini agar terekam pada video
	bannerColor := color.RGBA{R: 75, G: 85, B: 99, A: 255} // Abu-abu default
	if sp.isCooking {
		bannerColor = color.RGBA{R: 220, G: 38, B: 38, A: 255} // Merah
	} else if personInZone {
		bannerColor = color.RGBA{R: 217, G: 119, B: 6, A: 255} // Oranye/Kuning tua
	}
	
	// Gambar latar banner
	drawFilledRect(rgbaImg, 15, 15, 290, 48, bannerColor)
	// Gambar indikator ikon status (kotak putih di dalam banner)
	drawFilledRect(rgbaImg, 25, 25, 38, 38, color.White)

	// Re-encode ke JPEG
	var buf bytes.Buffer
	err = jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: 80})
	if err != nil {
		return jpegData, personInZone, maxConf
	}

	return buf.Bytes(), personInZone, maxConf
}

// saveCookingSnapshot menyimpan snapshot frame yang digambar box ke disk & log ke database
func (sp *StreamProcessor) saveCookingSnapshot(jpegData []byte, conf float64) {
	// Ambil frame terkini yang sudah digambar box-nya sebagai barang bukti
	sp.currentFrameMu.RLock()
	processedJPEG := make([]byte, len(sp.currentFrame))
	copy(processedJPEG, sp.currentFrame)
	sp.currentFrameMu.RUnlock()

	// Fallback jika buffer frame belum siap
	if len(processedJPEG) == 0 {
		processedJPEG = jpegData
	}

	// Nama file unik dengan timestamp
	snapshotFilename := fmt.Sprintf("cooking_%s.jpg", time.Now().Format("20060102_150405_000"))
	snapshotPath := filepath.Join("snapshots", snapshotFilename)

	// Pastikan folder snapshots ada
	os.MkdirAll("snapshots", 0755)

	err := os.WriteFile(snapshotPath, processedJPEG, 0644)
	if err != nil {
		log.Printf("Gagal menyimpan file snapshot memasak: %v", err)
		return
	}

	// Catat ke database SQLite
	_, dbErr := sp.db.LogDetection("cooking", conf, snapshotFilename)
	if dbErr != nil {
		log.Printf("Gagal mencatat log memasak ke SQLite: %v", dbErr)
	} else {
		log.Printf("DETEKSI MEMASAK AKTIF! Foto disimpan: %s", snapshotFilename)
	}
}

// drawBox menggambar garis batas kotak di gambar RGBA
func drawBox(img *image.RGBA, x1, y1, x2, y2 int, col color.Color, thickness int) {
	for t := 0; t < thickness; t++ {
		// Garis Atas
		for x := x1; x <= x2; x++ {
			img.Set(x, y1+t, col)
		}
		// Garis Bawah
		for x := x1; x <= x2; x++ {
			img.Set(x, y2-t, col)
		}
		// Garis Kiri
		for y := y1; y <= y2; y++ {
			img.Set(x1+t, y, col)
		}
		// Garis Kanan
		for y := y1; y <= y2; y++ {
			img.Set(x2-t, y, col)
		}
	}
}

// drawFilledRect menggambar kotak terisi penuh warna
func drawFilledRect(img *image.RGBA, x1, y1, x2, y2 int, col color.Color) {
	for y := y1; y <= y2; y++ {
		for x := x1; x <= x2; x++ {
			img.Set(x, y, col)
		}
	}
}
