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
func NewStreamProcessor(rtspURL string, aiClient *AIClient, db *DBManager, confThreshold float64) *StreamProcessor {
	return &StreamProcessor{
		rtspURL:       rtspURL,
		aiClient:      aiClient,
		db:            db,
		confThreshold: confThreshold,
		clients:       make(map[chan []byte]bool),
		stopChan:      make(chan struct{}),
		aiStatus:      "Unknown",
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
				// -rtsp_transport tcp: memaksa menggunakan TCP agar tidak ada frame drop
				cmd := exec.Command(ffmpegPath,
					"-rtsp_transport", "tcp",
					"-i", sp.rtspURL,
					"-f", "image2pipe",
					"-vcodec", "mjpeg",
					"-q:v", "4", // Kualitas gambar 1-31 (semakin kecil semakin bagus)
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

				// Scanner dengan buffer besar (maks 2MB) untuk mendukung resolusi gambar HD
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

					// Batasi pengiriman ke AI (misal 5 frame per detik / setiap 200ms)
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

					// Gambar bounding box pada gambar jika ada objek terdeteksi
					processedJPEG := jpegData
					var personDetected bool
					var maxConf float64

					// Hanya gambar box jika AI online dan mendeteksi sesuatu
					if aiErr == nil && len(detections) > 0 {
						processedJPEG, personDetected, maxConf = sp.drawBoundingBoxes(jpegData, detections)
						
						// Jika ada manusia terdeteksi dengan confidence di atas threshold, catat ke database
						if personDetected && maxConf >= sp.confThreshold {
							// Simpan snapshot
							snapshotFilename := fmt.Sprintf("person_%s.jpg", time.Now().Format("20060102_150405_000"))
							snapshotPath := filepath.Join("snapshots", snapshotFilename)
							
							// Pastikan folder snapshots ada
							os.MkdirAll("snapshots", 0755)
							
							err := os.WriteFile(snapshotPath, processedJPEG, 0644)
							if err == nil {
								// Log ke SQLite
								_, dbErr := sp.db.LogDetection("person", maxConf, snapshotFilename)
								if dbErr != nil {
									log.Printf("Gagal mencatat deteksi ke DB: %v", dbErr)
								}
							} else {
								log.Printf("Gagal menyimpan file snapshot: %v", err)
							}
						}
					} else if aiErr != nil && runAI {
						log.Printf("Layanan AI offline: %v", aiErr)
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

// drawBoundingBoxes menggambar kotak pembatas di frame JPEG jika terdeteksi manusia
func (sp *StreamProcessor) drawBoundingBoxes(jpegData []byte, detections []AIDetection) ([]byte, bool, float64) {
	// Decode gambar JPEG ke image.Image
	srcImg, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		return jpegData, false, 0
	}

	bounds := srcImg.Bounds()
	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, bounds, srcImg, bounds.Min, draw.Src)

	personDetected := false
	maxConf := 0.0

	// Definisikan warna-warna box
	colorRed := color.RGBA{R: 239, G: 68, B: 68, A: 255}   // Merah cerah untuk manusia
	colorYellow := color.RGBA{R: 245, G: 158, B: 11, A: 255} // Kuning untuk objek lain

	for _, det := range detections {
		// Filter deteksi: kita utamakan "person" (manusia)
		isPerson := det.Label == "person"
		
		if det.Confidence < sp.confThreshold {
			continue
		}

		if isPerson {
			personDetected = true
			if det.Confidence > maxConf {
				maxConf = det.Confidence
			}
		}

		// Koordinat box
		x1, y1 := int(det.Box[0]), int(det.Box[1])
		x2, y2 := int(det.Box[2]), int(det.Box[3])

		// Pastikan koordinat tidak melampaui batas gambar
		if x1 < 0 { x1 = 0 }
		if y1 < 0 { y1 = 0 }
		if x2 >= bounds.Max.X { x2 = bounds.Max.X - 1 }
		if y2 >= bounds.Max.Y { y2 = bounds.Max.Y - 1 }

		// Pilih warna berdasarkan jenis objek
		var boxColor color.Color = colorYellow
		if isPerson {
			boxColor = colorRed
		}

		// Gambar kotak pembatas dengan ketebalan 3 piksel
		drawBox(rgbaImg, x1, y1, x2, y2, boxColor, 3)
	}

	// Jika manusia terdeteksi, tambahkan banner ALERT di kiri atas gambar
	if personDetected {
		// Gambar banner latar belakang merah semi transparan
		// 15, 15 ke 250, 50
		bannerColor := color.RGBA{R: 220, G: 38, B: 38, A: 255}
		drawFilledRect(rgbaImg, 15, 15, 230, 48, bannerColor)
		
		// Gambar teks banner sederhana (simbol berupa kotak putih kecil)
		drawFilledRect(rgbaImg, 25, 25, 38, 38, color.White)
		// Kita tidak menggunakan parser font ttf agar program tetap ringan dan 100% andal di Windows.
		// Banner merah cerah sudah menjadi indikator visual yang sangat kuat.
	}

	// Re-encode ke JPEG
	var buf bytes.Buffer
	err = jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: 80})
	if err != nil {
		return jpegData, personDetected, maxConf
	}

	return buf.Bytes(), personDetected, maxConf
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
