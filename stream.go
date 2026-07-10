package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

	statusMu       sync.Mutex
	lastDetections []AIDetection
	detectionsMu   sync.Mutex
	aiBusy         bool
	aiBusyMu       sync.Mutex

	// Integrasi Gemini Hybrid AI
	geminiAPIKey       string
	geminiPrompt       string
	geminiDescription  string
	geminiFireAlert    bool
	geminiCookingAlert bool
	geminiLastCheck    time.Time
	geminiBusy         bool
	geminiBusyMu       sync.Mutex
}

// NewStreamProcessor membuat instansi baru StreamProcessor
func NewStreamProcessor(
	rtspURL string, 
	aiClient *AIClient, 
	db *DBManager, 
	confThreshold float64,
	zxMin, zyMin, zxMax, zyMax float64,
	cookingTriggerSecs int,
	geminiAPIKey string,
	geminiPrompt string,
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
		geminiAPIKey:       geminiAPIKey,
		geminiPrompt:       geminiPrompt,
		geminiDescription:  "Menunggu analisis pertama...",
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

				// Channel berkapasitas 1 (lossy channel)
				frameChan := make(chan []byte, 1)

				// Goroutine Pembaca Frame dari FFmpeg stdout secara asinkron
				go func() {
					scanner := bufio.NewScanner(stdout)
					buf := make([]byte, 1024*1024)
					scanner.Buffer(buf, 2*1024*1024)
					scanner.Split(splitJPEG)

					for scanner.Scan() {
						jpegData := scanner.Bytes()
						if len(jpegData) == 0 {
							continue
						}

						// Duplikasi byte frame karena buffer Scanner internal akan ditimpa
						frameCopy := make([]byte, len(jpegData))
						copy(frameCopy, jpegData)

						// Kirim ke channel secara non-blocking
						select {
						case frameChan <- frameCopy:
						default:
							// Buang frame lama jika channel penuh
							select {
							case <-frameChan:
							default:
							}
							frameChan <- frameCopy
						}
					}
					close(frameChan)
				}()

				lastAIDetectTime := time.Time{}
				frameCount := 0
				fpsLastTime := time.Now()

				// Bersihkan data deteksi lama saat memulai koneksi baru
				sp.detectionsMu.Lock()
				sp.lastDetections = nil
				sp.detectionsMu.Unlock()

				for jpegData := range frameChan {
					select {
					case <-sp.stopChan:
						cmd.Process.Kill()
						return
					default:
					}

					now := time.Now()

					// Hitung FPS visual stream
					frameCount++
					if frameCount >= 30 {
						elapsed := now.Sub(fpsLastTime).Seconds()
						sp.statusMu.Lock()
						sp.fps = float64(frameCount) / elapsed
						sp.statusMu.Unlock()
						frameCount = 0
						fpsLastTime = now
					}

					// Panggil AI secara asinkron (setiap 200ms jika sedang tidak sibuk)
					sp.aiBusyMu.Lock()
					busy := sp.aiBusy
					sp.aiBusyMu.Unlock()

					if now.Sub(lastAIDetectTime) >= 200*time.Millisecond && !busy {
						lastAIDetectTime = now

						sp.aiBusyMu.Lock()
						sp.aiBusy = true
						sp.aiBusyMu.Unlock()

						// Kirim deteksi ke background
						go func(img []byte) {
							aiStart := time.Now()
							detections, aiErr := sp.aiClient.Detect(img)
							latency := time.Since(aiStart)

							sp.aiBusyMu.Lock()
							sp.aiBusy = false
							sp.aiBusyMu.Unlock()

							sp.statusMu.Lock()
							sp.aiLatency = latency
							if aiErr != nil {
								sp.aiStatus = "Offline"
								// Kosongkan deteksi jika offline agar kotak tidak melayang selamanya
								sp.detectionsMu.Lock()
								sp.lastDetections = nil
								sp.detectionsMu.Unlock()
							} else {
								sp.aiStatus = "Online"
								sp.detectionsMu.Lock()
								sp.lastDetections = detections
								sp.detectionsMu.Unlock()
							}
							sp.statusMu.Unlock()
						}(jpegData)
					}

					// Panggil Google Gemini API kognitif secara asinkron (setiap 15 detik jika sedang tidak sibuk)
					sp.geminiBusyMu.Lock()
					gBusy := sp.geminiBusy
					sp.geminiBusyMu.Unlock()

					if sp.geminiAPIKey != "" && now.Sub(sp.geminiLastCheck) >= 15*time.Second && !gBusy {
						sp.geminiLastCheck = now
						sp.callGeminiAPI(jpegData)
					}

					// Salin deteksi terakhir untuk digambar pada frame ini
					sp.detectionsMu.Lock()
					var detections []AIDetection
					if sp.lastDetections != nil {
						detections = make([]AIDetection, len(sp.lastDetections))
						copy(detections, sp.lastDetections)
					}
					sp.detectionsMu.Unlock()

					// Proses logika deteksi ROI dapur & penggambaran bounding box
					processedJPEG, _, _ := sp.drawBoundingBoxes(jpegData, detections)

					// Simpan frame terbaru untuk klien baru
					sp.currentFrameMu.Lock()
					sp.currentFrame = processedJPEG
					sp.currentFrameMu.Unlock()

					// Siarkan ke seluruh klien browser yang terhubung
					sp.broadcast(processedJPEG)
				}

				if err := cmd.Wait(); err != nil {
					log.Printf("Proses FFmpeg berhenti: %v", err)
				}
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

	localFireDetected := false
	localSmokeDetected := false

	// Proses setiap objek terdeteksi
	for _, det := range detections {
		isPerson := det.Label == "person"
		isFire := det.Label == "fire"
		isSmoke := det.Label == "smoke"

		// Kita memproses manusia, api, atau asap
		if !isPerson && !isFire && !isSmoke {
			continue
		}
		
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

		// Pilih warna dan label teks berdasarkan jenis objek
		var boxColor color.Color = colorRed
		var labelText string

		if isPerson {
			boxColor = colorRed
			confPercent := int(det.Confidence * 100)
			labelText = fmt.Sprintf("Manusia %d%%", confPercent)

			// LOGIKA DETEKSI ZONA MEMASAK (ROI):
			// Kita ambil posisi kaki orang tersebut (tengah bawah dari bounding box):
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
		} else if isFire {
			boxColor = color.RGBA{R: 249, G: 115, B: 22, A: 255} // Orange menyala
			confPercent := int(det.Confidence * 100)
			labelText = fmt.Sprintf("API! %d%%", confPercent)
			localFireDetected = true
		} else if isSmoke {
			boxColor = color.RGBA{R: 156, G: 163, B: 175, A: 255} // Abu-abu asap
			confPercent := int(det.Confidence * 100)
			labelText = fmt.Sprintf("ASAP! %d%%", confPercent)
			localSmokeDetected = true
		}

		// Gambar kotak pembatas objek
		drawBox(rgbaImg, x1, y1, x2, y2, boxColor, 3)

		// Gambar label teks di atas kotak
		fontScale := 1
		if height > 500 {
			fontScale = 2
		}
		charStep := 7 * fontScale
		labelHeight := 8 * fontScale

		labelY := y1 - labelHeight - 4
		if labelY < 0 {
			labelY = y1 + 5
		}

		labelWidth := len(labelText) * charStep
		drawFilledRect(rgbaImg, x1, labelY, x1+labelWidth, labelY+labelHeight+2, boxColor)
		drawText(rgbaImg, x1+3, labelY+2, labelText, color.White, fontScale)
	}

	// Update alarm status lokal berdasarkan deteksi sensor visual YOLOv8
	sp.statusMu.Lock()
	if localFireDetected || localSmokeDetected {
		if !sp.geminiFireAlert {
			sp.geminiFireAlert = true
			if localFireDetected {
				sp.geminiDescription = "DETEKSI DARURAT LOKAL (YOLOv8): Terdeteksi indikasi API secara real-time!"
			} else {
				sp.geminiDescription = "DETEKSI DARURAT LOKAL (YOLOv8): Terdeteksi indikasi ASAP secara real-time!"
			}
			log.Println("WARNING!!! DETEKSI BAHAYA LOKAL AKTIF!")
			go sp.saveFireSnapshot(jpegData)
		}
	} else if sp.geminiAPIKey == "" {
		// Hanya matikan alarm secara otomatis jika Gemini nonaktif (agar tidak menimpa status Gemini jika aktif)
		if sp.geminiFireAlert {
			sp.geminiFireAlert = false
			sp.geminiDescription = "Sistem lokal berjalan normal. Tidak terdeteksi manusia, api, atau asap."
		}
	}
	sp.statusMu.Unlock()

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

// Tiny font 8x8 bitmap untuk set karakter: "Manusia0123456789% ."
var font8x8 = map[rune][8]byte{
	'M': {0x81, 0xC3, 0xA5, 0x99, 0x81, 0x81, 0x81, 0x00},
	'a': {0x00, 0x00, 0x7C, 0x02, 0x7E, 0x44, 0x4A, 0x3E},
	'n': {0x00, 0x00, 0x74, 0x4A, 0x42, 0x42, 0x42, 0x00},
	'u': {0x00, 0x00, 0x42, 0x42, 0x42, 0x46, 0x3A, 0x00},
	's': {0x00, 0x00, 0x3E, 0x40, 0x3E, 0x02, 0x3E, 0x00},
	'i': {0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x1C, 0x00},
	'0': {0x3C, 0x46, 0x4A, 0x52, 0x62, 0x62, 0x3C, 0x00},
	'1': {0x18, 0x28, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00},
	'2': {0x3C, 0x42, 0x02, 0x3C, 0x40, 0x40, 0x7E, 0x00},
	'3': {0x3C, 0x42, 0x0C, 0x02, 0x02, 0x42, 0x3C, 0x00},
	'4': {0x08, 0x18, 0x28, 0x48, 0x7E, 0x08, 0x08, 0x00},
	'5': {0x7E, 0x40, 0x7C, 0x02, 0x02, 0x42, 0x3C, 0x00},
	'6': {0x3C, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x3C, 0x00},
	'7': {0x7E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10, 0x00},
	'8': {0x3C, 0x42, 0x42, 0x3C, 0x42, 0x42, 0x3C, 0x00},
	'9': {0x3C, 0x42, 0x42, 0x3E, 0x02, 0x02, 0x3C, 0x00},
	'%': {0x42, 0x24, 0x14, 0x08, 0x14, 0x24, 0x42, 0x00},
	' ': {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
	'.': {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18},
}

// drawText menggambar teks pada gambar RGBA menggunakan skala pembesaran pixel (fontScale)
func drawText(img *image.RGBA, x, y int, text string, col color.Color, scale int) {
	for _, char := range text {
		bitmap, found := font8x8[char]
		if !found {
			bitmap = font8x8[' ']
		}

		for row := 0; row < 8; row++ {
			b := bitmap[row]
			for colIdx := 0; colIdx < 8; colIdx++ {
				if (b & (1 << (7 - colIdx))) != 0 {
					for dy := 0; dy < scale; dy++ {
						for dx := 0; dx < scale; dx++ {
							px := x + colIdx*scale + dx
							py := y + row*scale + dy
							if px >= 0 && px < img.Bounds().Max.X && py >= 0 && py < img.Bounds().Max.Y {
								img.Set(px, py, col)
							}
						}
					}
				}
			}
		}
		x += 7 * scale // Geser posisi x untuk huruf berikutnya
	}
}

// ============================================================================
// INTEGRASI GOOGLE GEMINI HYBRID AI (VLM)
// ============================================================================

// Struktur Request payload untuk Gemini API
type GeminiRequest struct {
	Contents         []GeminiContent  `json:"contents"`
	GenerationConfig *GeminiGenConfig `json:"generationConfig,omitempty"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *GeminiInlineData `json:"inlineData,omitempty"`
}

type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64
}

type GeminiGenConfig struct {
	ResponseMimeType string `json:"responseMimeType"`
}

// Struktur Response payload dari Gemini API
type GeminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// Hasil analisis yang diharapkan dalam format JSON
type GeminiAnalysisResult struct {
	Cooking     bool   `json:"cooking"`
	Fire        bool   `json:"fire"`
	Description string `json:"description"`
}

// callGeminiAPI mengirimkan frame JPEG ke Google Gemini API secara asinkron (background)
func (sp *StreamProcessor) callGeminiAPI(jpegData []byte) {
	if sp.geminiAPIKey == "" {
		return
	}

	// Set status busy
	sp.geminiBusyMu.Lock()
	sp.geminiBusy = true
	sp.geminiBusyMu.Unlock()

	go func() {
		defer func() {
			sp.geminiBusyMu.Lock()
			sp.geminiBusy = false
			sp.geminiBusyMu.Unlock()
		}()

		// Encode frame JPEG ke Base64
		base64Data := base64.StdEncoding.EncodeToString(jpegData)

		// Bentuk payload request sesuai format Gemini API
		reqPayload := GeminiRequest{
			Contents: []GeminiContent{
				{
					Parts: []GeminiPart{
						{
							Text: sp.geminiPrompt,
						},
						{
							InlineData: &GeminiInlineData{
								MimeType: "image/jpeg",
								Data:     base64Data,
							},
						},
					},
				},
			},
			GenerationConfig: &GeminiGenConfig{
				ResponseMimeType: "application/json",
			},
		}

		// Marshal ke JSON
		jsonBytes, err := json.Marshal(reqPayload)
		if err != nil {
			log.Printf("Gagal marshal request payload Gemini: %v", err)
			sp.statusMu.Lock()
			sp.geminiDescription = fmt.Sprintf("Error: Gagal menyusun data request (%v)", err)
			sp.statusMu.Unlock()
			return
		}

		// URL endpoint Gemini API (menggunakan model gemini-flash-latest)
		url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=%s", sp.geminiAPIKey)

		// Buat HTTP Request
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
		if err != nil {
			log.Printf("Gagal membuat HTTP request Gemini: %v", err)
			sp.statusMu.Lock()
			sp.geminiDescription = fmt.Sprintf("Error: Gagal inisialisasi request HTTP (%v)", err)
			sp.statusMu.Unlock()
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-goog-api-key", sp.geminiAPIKey) // Header penting untuk token API bertipe tertentu

		// Eksekusi HTTP Request
		client := &http.Client{Timeout: 15 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Gagal menghubungi Gemini API: %v", err)
			sp.statusMu.Lock()
			sp.geminiDescription = fmt.Sprintf("Error: Tidak ada koneksi internet / gagal menghubungi Gemini API (%v)", err)
			sp.statusMu.Unlock()
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			log.Printf("Gemini API mengembalikan status error: %d, Response: %s", resp.StatusCode, string(bodyBytes))
			sp.statusMu.Lock()
			if resp.StatusCode == http.StatusTooManyRequests {
				sp.geminiDescription = "Error: Limit kuota API terlampaui (HTTP 429: Too Many Requests). Menunggu giliran berikutnya secara otomatis..."
			} else {
				sp.geminiDescription = fmt.Sprintf("Error: API mengembalikan kode %d (Periksa validitas Gemini API Key Anda)", resp.StatusCode)
			}
			sp.statusMu.Unlock()
			return
		}

		// Decode Response
		var geminiResp GeminiResponse
		if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
			log.Printf("Gagal decode response Gemini API: %v", err)
			sp.statusMu.Lock()
			sp.geminiDescription = fmt.Sprintf("Error: Gagal memproses data respons server (%v)", err)
			sp.statusMu.Unlock()
			return
		}

		if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
			log.Printf("Gemini API tidak mengembalikan konten candidates.")
			sp.statusMu.Lock()
			sp.geminiDescription = "Error: Google Gemini mengembalikan hasil kosong (Candidates Empty)"
			sp.statusMu.Unlock()
			return
		}

		// Parsing JSON teks hasil generasi Gemini (Sanitasi jika ada wrapper markdown)
		rawText := geminiResp.Candidates[0].Content.Parts[0].Text
		cleanJSON := sanitizeJSON(rawText)

		var result GeminiAnalysisResult
		if err := json.Unmarshal([]byte(cleanJSON), &result); err != nil {
			log.Printf("Gagal unmarshal hasil teks Gemini ke JSON: %v. Raw text: %s", err, rawText)
			sp.statusMu.Lock()
			sp.geminiDescription = fmt.Sprintf("Error: Gagal membaca format teks kognitif AI (%v)", err)
			sp.statusMu.Unlock()
			return
		}

		// Simpan hasil ke state StreamProcessor secara thread-safe
		sp.statusMu.Lock()
		sp.geminiDescription = result.Description
		sp.geminiCookingAlert = result.Cooking
		
		// Jika terjadi transisi alarm kebakaran (sebelumnya aman sekarang terdeteksi api)
		fireTriggered := result.Fire && !sp.geminiFireAlert
		sp.geminiFireAlert = result.Fire
		sp.statusMu.Unlock()

		// Tangani alarm kebakaran aktif
		if fireTriggered {
			log.Printf("WARNING!!! DETEKSI KEBAKARAN/API DARI GEMINI: %s", result.Description)
			// Simpan bukti foto snapshot kebakaran
			go sp.saveFireSnapshot(jpegData)
		}
	}()
}

// sanitizeJSON membersihkan format JSON mentah dari blok tag markdown kustom (seperti ```json ... ```)
func sanitizeJSON(input string) string {
	start := strings.Index(input, "{")
	end := strings.LastIndex(input, "}")
	if start == -1 || end == -1 || start >= end {
		return input
	}
	return input[start : end+1]
}

// saveFireSnapshot menyimpan snapshot bukti kebakaran ke disk & log ke database SQLite
func (sp *StreamProcessor) saveFireSnapshot(jpegData []byte) {
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
	snapshotFilename := fmt.Sprintf("fire_%s.jpg", time.Now().Format("20060102_150405_000"))
	snapshotPath := filepath.Join("snapshots", snapshotFilename)

	// Pastikan folder snapshots ada
	os.MkdirAll("snapshots", 0755)

	err := os.WriteFile(snapshotPath, processedJPEG, 0644)
	if err != nil {
		log.Printf("Gagal menyimpan file snapshot kebakaran: %v", err)
		return
	}

	// Catat alarm kebakaran ke database SQLite
	_, dbErr := sp.db.LogDetection("fire", 0.99, snapshotFilename) // Default confidence 99%
	if dbErr != nil {
		log.Printf("Gagal mencatat log kebakaran ke SQLite: %v", dbErr)
	} else {
		log.Printf("DARURAT KEBAKARAN TERCATAT! Foto disimpan: %s", snapshotFilename)
	}
}


