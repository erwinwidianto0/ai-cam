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
	rtspURL          string
	aiClient         *AIClient
	db               *DBManager
	confThreshold    float64
	streamResolution string

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

	// Integrasi Gemini/OpenAI Hybrid AI
	vlmProvider         string
	openaiAPIKey        string
	geminiAPIKey        string
	geminiPrompt        string
	geminiDescription   string
	geminiFireAlert         bool
	geminiCookingAlert      bool
	geminiSmokingAlert      bool
	geminiSleepingAlert     bool
	geminiSOPViolationAlert bool
	geminiSmokingCount      int
	geminiSleepingCount     int
	geminiSOPViolationCount int
	geminiLastCheck         time.Time
	geminiBusy              bool
	geminiBusyMu            sync.Mutex
}

// NewStreamProcessor membuat instansi baru StreamProcessor
func NewStreamProcessor(
	rtspURL string, 
	aiClient *AIClient, 
	db *DBManager, 
	confThreshold float64,
	zxMin, zyMin, zxMax, zyMax float64,
	cookingTriggerSecs int,
	vlmProvider string,
	openaiAPIKey string,
	geminiAPIKey string,
	geminiPrompt string,
	streamResolution string,
) *StreamProcessor {
	return &StreamProcessor{
		rtspURL:            rtspURL,
		aiClient:           aiClient,
		db:                 db,
		confThreshold:      confThreshold,
		streamResolution:   streamResolution,
		zoneXMin:           zxMin,
		zoneYMin:           zyMin,
		zoneXMax:           zxMax,
		zoneYMax:           zyMax,
		cookingTriggerSecs: cookingTriggerSecs,
		vlmProvider:        vlmProvider,
		openaiAPIKey:       openaiAPIKey,
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
				
				// Argumen ffmpeg untuk mengambil stream secara dinamis
				var args []string
				isRTSP := strings.HasPrefix(strings.ToLower(sp.rtspURL), "rtsp://")
				isHLS := strings.Contains(strings.ToLower(sp.rtspURL), ".m3u8")
				
				if isRTSP {
					args = []string{
						"-fflags", "nobuffer",       // Matikan buffering input jaringan
						"-flags", "low_delay",       // Paksa mode delay rendah
						"-analyzeduration", "0",     // Jangan buang waktu menganalisis format
						"-probesize", "32",          // Ukuran probe minimal
						"-rtsp_transport", "tcp",
						"-i", sp.rtspURL,
					}
				} else if isHLS {
					args = []string{
						"-fflags", "nobuffer",       // Matikan buffering input jaringan
						"-flags", "low_delay",       // Paksa mode delay rendah
						"-i", sp.rtspURL,
					}
				} else {
					args = []string{
						"-fflags", "nobuffer",
						"-flags", "low_delay",
						"-re",
						"-stream_loop", "-1",
						"-i", sp.rtspURL,
					}
				}

				// Tambahkan filter skala resolusi secara dinamis jika tidak diset "original"
				if sp.streamResolution != "original" && sp.streamResolution != "" {
					resStr := strings.Replace(sp.streamResolution, "x", ":", 1)
					args = append(args, "-vf", fmt.Sprintf("scale=%s", resStr))
				}

				// Tambahkan parameter output MJPEG
				args = append(args,
					"-r", "30",                  // Paksa output stabil di 30 FPS
					"-f", "image2pipe",
					"-vcodec", "mjpeg",
					"-q:v", "4",
					"-pix_fmt", "yuvj420p",
					"-",
				)
				cmd := exec.Command(ffmpegPath, args...)

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

					// Salin deteksi terakhir untuk analisis ROI dan VLM
					sp.detectionsMu.Lock()
					var detections []AIDetection
					if sp.lastDetections != nil {
						detections = make([]AIDetection, len(sp.lastDetections))
						copy(detections, sp.lastDetections)
					}
					sp.detectionsMu.Unlock()

					// Jika ada alarm aktif (kebakaran, merokok, tidur, pelanggaran SOP) atau aktivitas memasak sedang berjalan, 
					// lakukan pemantauan berkala ke VLM setiap 20 detik agar alarm bisa mati secara otomatis jika kondisi sudah kembali aman.
					sp.statusMu.Lock()
					hasActiveAlert := sp.geminiFireAlert || sp.geminiSmokingAlert || sp.geminiSleepingAlert || sp.geminiSOPViolationAlert || sp.isCooking
					sp.statusMu.Unlock()

					// Cari apakah ada manusia terdeteksi secara lokal oleh YOLO
					localPersonDetected := false
					for _, d := range detections {
						if d.Label == "person" && d.Confidence >= sp.confThreshold {
							localPersonDetected = true
							break
						}
					}

					sp.geminiBusyMu.Lock()
					gBusy := sp.geminiBusy
					sp.geminiBusyMu.Unlock()

					hasAPIKey := false
					if sp.vlmProvider == "openai" {
						hasAPIKey = (sp.openaiAPIKey != "")
					} else {
						hasAPIKey = (sp.geminiAPIKey != "")
					}

					// Pemicu VLM periodik:
					// 1. Jika ada alarm aktif: cek setiap 20 detik untuk melihat apakah kondisi sudah aman kembali.
					// 2. Jika tidak ada alarm aktif, tetapi ada orang (person) di area kerja: cek setiap 60 detik (1 menit) untuk mendeteksi rokok/tidur/pelanggaran secara otomatis.
					shouldCheckVLM := false
					if hasActiveAlert {
						shouldCheckVLM = time.Since(sp.geminiLastCheck) >= 20*time.Second
					} else if localPersonDetected {
						shouldCheckVLM = time.Since(sp.geminiLastCheck) >= 60*time.Second
					}

					if hasAPIKey && shouldCheckVLM && !gBusy {
						sp.geminiLastCheck = time.Now()
						go sp.callVLMAPI(jpegData)
					}

					// 1. Dapatkan dimensi JPEG tanpa decode pixel (sangat cepat, 0.001ms)
					var width, height float64 = 640, 360
					if cfg, _, err := image.DecodeConfig(bytes.NewReader(jpegData)); err == nil {
						width = float64(cfg.Width)
						height = float64(cfg.Height)
					}

					// 2. Hitung status memasak dan zona kompor secara lokal
					personInZone, maxConf := sp.checkPersonInZone(detections, width, height)
					localFireDetected, localSmokeDetected := false, false
					for _, d := range detections {
						if d.Confidence >= sp.confThreshold {
							if d.Label == "fire" {
								localFireDetected = true
							} else if d.Label == "smoke" {
								localSmokeDetected = true
							}
						}
					}

					// 3. Update alarm status lokal (api/asap)
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
							
							// Gambar box khusus pada snapshot bukti kebakaran sebelum disimpan ke disk
							go sp.saveFireSnapshot(jpegData, detections)

							sp.geminiBusyMu.Lock()
							gBusy := sp.geminiBusy
							sp.geminiBusyMu.Unlock()
							if hasAPIKey && !gBusy && time.Since(sp.geminiLastCheck) >= 10*time.Second {
								sp.geminiLastCheck = time.Now()
								go sp.callVLMAPI(jpegData)
							}
						}
					} else {
						if sp.geminiFireAlert {
							sp.geminiFireAlert = false
						}
						if !sp.isCooking {
							if localPersonDetected {
								sp.geminiDescription = "Sistem lokal berjalan normal. Terdeteksi manusia, area aman dari api dan asap."
							} else {
								sp.geminiDescription = "Sistem lokal berjalan normal. Area aman (tidak terdeteksi manusia, api, atau asap)."
							}
						}
					}
					sp.statusMu.Unlock()

					// 4. State Machine Deteksi Memasak
					if personInZone {
						if sp.inZoneStartTime.IsZero() {
							sp.inZoneStartTime = now
						}
						sp.lastSeenInZone = now
						
						elapsed := now.Sub(sp.inZoneStartTime)
						sp.secondsInZone = int(elapsed.Seconds())
						
						if sp.secondsInZone >= sp.cookingTriggerSecs {
							if !sp.isCooking {
								sp.isCooking = true
								sp.kitchenStatus = "Memasak"
								
								// Gambar box pada snapshot bukti memasak sebelum disimpan ke disk
								go sp.saveCookingSnapshot(jpegData, maxConf, detections)

								sp.geminiBusyMu.Lock()
								gBusy := sp.geminiBusy
								sp.geminiBusyMu.Unlock()
								if hasAPIKey && !gBusy && time.Since(sp.geminiLastCheck) >= 10*time.Second {
									sp.geminiLastCheck = time.Now()
									go sp.callVLMAPI(jpegData)
								}
							}
						}
					} else {
						if !sp.inZoneStartTime.IsZero() {
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

					// Salin frame mentah ke currentFrame untuk klien baru
					sp.currentFrameMu.Lock()
					sp.currentFrame = jpegData
					sp.currentFrameMu.Unlock()

					// Siarkan frame mentah langsung ke browser (Sangat cepat! Mengalir di 30 FPS penuh!)
					sp.broadcast(jpegData)
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

// checkPersonInZone menganalisis apakah ada orang di ROI berdasarkan bounding box tanpa perlu decode piksel
func (sp *StreamProcessor) checkPersonInZone(detections []AIDetection, width, height float64) (bool, float64) {
	personInZone := false
	maxConf := 0.0
	for _, det := range detections {
		if det.Label == "person" && det.Confidence >= sp.confThreshold {
			x1, _ := det.Box[0], det.Box[1]
			x2, y2 := det.Box[2], det.Box[3]

			px := (x1 + x2) / 2
			py := y2

			normX := px / width
			normY := py / height

			if normX >= sp.zoneXMin && normX <= sp.zoneXMax && normY >= sp.zoneYMin && normY <= sp.zoneYMax {
				personInZone = true
				if det.Confidence > maxConf {
					maxConf = det.Confidence
				}
			}
		}
	}
	return personInZone, maxConf
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

	personInZone := false
	maxConf := 0.0
	localPersonDetected := false
	localFireDetected := false
	localSmokeDetected := false

	// Cek apakah gambar bertipe YCbCr (format asli decoder JPEG Go untuk MJPEG stream)
	ycbcrImg, isYCbCr := srcImg.(*image.YCbCr)
	if isYCbCr {
		// ====================================================================
		// JALUR SUPER-CEPAT: Gambar langsung pada YCbCr (Tanpa Alokasi RGBA / Tanpa Copy Pixel)
		// ====================================================================

		// Proses setiap objek terdeteksi
		for _, det := range detections {
			isPerson := det.Label == "person"
			isFire := det.Label == "fire"
			isSmoke := det.Label == "smoke"

			if det.Confidence < sp.confThreshold {
				continue
			}

			x1, y1 := int(det.Box[0]), int(det.Box[1])
			x2, y2 := int(det.Box[2]), int(det.Box[3])

			if x1 < 0 { x1 = 0 }
			if y1 < 0 { y1 = 0 }
			if x2 >= bounds.Max.X { x2 = bounds.Max.X - 1 }
			if y2 >= bounds.Max.Y { y2 = bounds.Max.Y - 1 }

			var dy, dcb, dcr uint8
			var labelText string

			if isPerson {
				localPersonDetected = true
				dy, dcb, dcr = 92, 97, 219 // Red
				confPercent := int(det.Confidence * 100)
				labelText = fmt.Sprintf("Manusia %d%%", confPercent)

				px := (x1 + x2) / 2
				py := y2
				normX := float64(px) / width
				normY := float64(py) / height

				if normX >= sp.zoneXMin && normX <= sp.zoneXMax && normY >= sp.zoneYMin && normY <= sp.zoneYMax {
					personInZone = true
					if det.Confidence > maxConf {
						maxConf = det.Confidence
					}
				}
			} else if isFire {
				dy, dcb, dcr = 167, 40, 183 // Orange/Yellow
				confPercent := int(det.Confidence * 100)
				labelText = fmt.Sprintf("API! %d%%", confPercent)
				localFireDetected = true
			} else if isSmoke {
				dy, dcb, dcr = 128, 128, 128 // Gray
				confPercent := int(det.Confidence * 100)
				labelText = fmt.Sprintf("ASAP! %d%%", confPercent)
				localSmokeDetected = true
			} else {
				dy, dcb, dcr = 128, 128, 48 // Green
				confPercent := int(det.Confidence * 100)
				labelText = fmt.Sprintf("%s %d%%", det.Label, confPercent)
			}

			drawBoxYCbCr(ycbcrImg, x1, y1, x2, y2, dy, dcb, dcr, 3)

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
			drawFilledRectYCbCr(ycbcrImg, x1, labelY, x1+labelWidth, labelY+labelHeight+2, dy, dcb, dcr)
			drawTextYCbCr(ycbcrImg, x1+3, labelY+2, labelText, 255, 128, 128, fontScale) // White text
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
				go sp.saveFireSnapshot(jpegData, detections)

				hasAPIKey := false
				if sp.vlmProvider == "openai" {
					hasAPIKey = (sp.openaiAPIKey != "")
				} else {
					hasAPIKey = (sp.geminiAPIKey != "")
				}

				sp.geminiBusyMu.Lock()
				gBusy := sp.geminiBusy
				sp.geminiBusyMu.Unlock()
				if hasAPIKey && !gBusy && time.Since(sp.geminiLastCheck) >= 10*time.Second {
					sp.geminiLastCheck = time.Now()
					go sp.callVLMAPI(jpegData)
				}
			}
		} else {
			if sp.geminiFireAlert {
				sp.geminiFireAlert = false
			}
			if !sp.isCooking {
				if localPersonDetected {
					sp.geminiDescription = "Sistem lokal berjalan normal. Terdeteksi manusia, area aman dari api dan asap."
				} else {
					sp.geminiDescription = "Sistem lokal berjalan normal. Area aman (tidak terdeteksi manusia, api, atau asap)."
				}
			}
		}
		sp.statusMu.Unlock()

		// Gambar Zona Kompor (ROI)
		zx1 := int(sp.zoneXMin * width)
		zy1 := int(sp.zoneYMin * height)
		zx2 := int(sp.zoneXMax * width)
		zy2 := int(sp.zoneYMax * height)

		var zxY, zxCb, zxCr uint8 = 122, 198, 83 // Blue
		if sp.isCooking {
			zxY, zxCb, zxCr = 92, 97, 219 // Red
		} else if personInZone {
			zxY, zxCb, zxCr = 167, 40, 183 // Yellow
		}

		drawBoxYCbCr(ycbcrImg, zx1, zy1, zx2, zy2, zxY, zxCb, zxCr, 2)
		drawFilledRectYCbCr(ycbcrImg, zx1, zy1, zx1+100, zy1+18, zxY, zxCb, zxCr)
		drawTextYCbCr(ycbcrImg, zx1+3, zy1+3, "ZONA KOMPOR", 255, 128, 128, 1)

		// Gambar banner status dapur di kiri atas
		var bY, bCb, bCr uint8 = 128, 128, 128 // Gray
		if sp.isCooking {
			bY, bCb, bCr = 92, 97, 219 // Red
		} else if personInZone {
			bY, bCb, bCr = 167, 40, 183 // Orange/Yellow
		}

		drawFilledRectYCbCr(ycbcrImg, 15, 15, 290, 48, bY, bCb, bCr)
		drawFilledRectYCbCr(ycbcrImg, 25, 25, 38, 38, 255, 128, 128) // White icon block

		var buf bytes.Buffer
		err = jpeg.Encode(&buf, ycbcrImg, &jpeg.Options{Quality: 80})
		if err != nil {
			return jpegData, personInZone, maxConf
		}
		return buf.Bytes(), personInZone, maxConf
	}

	// ====================================================================
	// JALUR FALLBACK: Menggunakan RGBA (Jika decoder menghasilkan non-YCbCr)
	// ====================================================================
	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, bounds, srcImg, bounds.Min, draw.Src)

	// Definisikan warna-warna box
	colorRed := color.RGBA{R: 239, G: 68, B: 68, A: 255}   // Merah cerah
	colorYellow := color.RGBA{R: 245, G: 158, B: 11, A: 255} // Kuning
	colorBlue := color.RGBA{R: 59, G: 130, B: 246, A: 255}   // Biru

	// Proses setiap objek terdeteksi
	for _, det := range detections {
		isPerson := det.Label == "person"
		isFire := det.Label == "fire"
		isSmoke := det.Label == "smoke"

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
			localPersonDetected = true
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
		} else {
			// Kelas kustom lainnya (mobil, motor, pemadam api, dll)
			boxColor = color.RGBA{R: 16, G: 185, B: 129, A: 255} // Hijau toska
			confPercent := int(det.Confidence * 100)
			labelText = fmt.Sprintf("%s %d%%", det.Label, confPercent)
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
			go sp.saveFireSnapshot(jpegData, detections)

			// Cek kunci API sesuai provider VLM terpilih
			hasAPIKey := false
			if sp.vlmProvider == "openai" {
				hasAPIKey = (sp.openaiAPIKey != "")
			} else {
				hasAPIKey = (sp.geminiAPIKey != "")
			}

			// Panggil VLM (Teacher) untuk memverifikasi bahaya kebakaran yang dideteksi oleh YOLO
			sp.geminiBusyMu.Lock()
			gBusy := sp.geminiBusy
			sp.geminiBusyMu.Unlock()
			if hasAPIKey && !gBusy && time.Since(sp.geminiLastCheck) >= 10*time.Second {
				sp.geminiLastCheck = time.Now()
				go sp.callVLMAPI(jpegData)
			}
		}
	} else {
		if sp.geminiFireAlert {
			sp.geminiFireAlert = false
		}
		
		// Jika tidak sedang memasak dan tidak ada bahaya kebakaran, gunakan deskripsi lokal YOLO (100% hemat token)
		if !sp.isCooking {
			if localPersonDetected {
				sp.geminiDescription = "Sistem lokal berjalan normal. Terdeteksi manusia, area aman dari api dan asap."
			} else {
				sp.geminiDescription = "Sistem lokal berjalan normal. Area aman (tidak terdeteksi manusia, api, atau asap)."
			}
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
				go sp.saveCookingSnapshot(jpegData, maxConf, detections)

				// Cek kunci API sesuai provider VLM terpilih
				hasAPIKey := false
				if sp.vlmProvider == "openai" {
					hasAPIKey = (sp.openaiAPIKey != "")
				} else {
					hasAPIKey = (sp.geminiAPIKey != "")
				}

				// Panggil VLM (Teacher) sekali saja ketika transisi memasak baru dimulai untuk analisis kognitif cerdas
				sp.geminiBusyMu.Lock()
				gBusy := sp.geminiBusy
				sp.geminiBusyMu.Unlock()
				if hasAPIKey && !gBusy && time.Since(sp.geminiLastCheck) >= 10*time.Second {
					sp.geminiLastCheck = time.Now()
					go sp.callVLMAPI(jpegData)
				}
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
func (sp *StreamProcessor) saveCookingSnapshot(jpegData []byte, conf float64, detections []AIDetection) {
	// Gambar box terlebih dahulu sebelum disimpan (hanya sekali, jadi tidak membebani CPU)
	processedJPEG, _, _ := sp.drawBoundingBoxes(jpegData, detections)

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
		// Simpan otomatis foto bersih dan anotasi koordinat YOLO ke dataset latihan
		go sp.saveToTrainingDataset(jpegData, detections)
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

// YCbCr helper functions for super-fast direct drawing (saving 80% CPU on high resolutions)
func drawPixelYCbCr(img *image.YCbCr, x, y int, yVal, cbVal, crVal uint8) {
	if x < img.Rect.Min.X || x >= img.Rect.Max.X || y < img.Rect.Min.Y || y >= img.Rect.Max.Y {
		return
	}
	yi := img.YOffset(x, y)
	ci := img.COffset(x, y)
	img.Y[yi] = yVal
	img.Cb[ci] = cbVal
	img.Cr[ci] = crVal
}

func drawBoxYCbCr(img *image.YCbCr, x1, y1, x2, y2 int, yVal, cbVal, crVal uint8, thickness int) {
	for t := 0; t < thickness; t++ {
		// Garis horizontal atas & bawah
		for x := x1; x <= x2; x++ {
			drawPixelYCbCr(img, x, y1+t, yVal, cbVal, crVal)
			drawPixelYCbCr(img, x, y2-t, yVal, cbVal, crVal)
		}
		// Garis vertikal kiri & kanan
		for y := y1; y <= y2; y++ {
			drawPixelYCbCr(img, x1+t, y, yVal, cbVal, crVal)
			drawPixelYCbCr(img, x2-t, y, yVal, cbVal, crVal)
		}
	}
}

func drawFilledRectYCbCr(img *image.YCbCr, x1, y1, x2, y2 int, yVal, cbVal, crVal uint8) {
	for y := y1; y <= y2; y++ {
		for x := x1; x <= x2; x++ {
			drawPixelYCbCr(img, x, y, yVal, cbVal, crVal)
		}
	}
}

func drawTextYCbCr(img *image.YCbCr, x, y int, text string, yVal, cbVal, crVal uint8, scale int) {
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
							drawPixelYCbCr(img, x+colIdx*scale+dx, y+row*scale+dy, yVal, cbVal, crVal)
						}
					}
				}
			}
		}
		x += 7 * scale
	}
}

// Tiny font 8x8 bitmap untuk set karakter lengkap (A-Z, a-z, digit, tanda baca)
var font8x8 = map[rune][8]byte{
	'A': {0x18, 0x24, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x00},
	'B': {0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x7C, 0x00},
	'C': {0x3C, 0x42, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00},
	'D': {0x78, 0x44, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00},
	'E': {0x7E, 0x40, 0x40, 0x78, 0x40, 0x40, 0x7E, 0x00},
	'F': {0x7E, 0x40, 0x40, 0x78, 0x40, 0x40, 0x40, 0x00},
	'G': {0x3C, 0x42, 0x40, 0x4E, 0x42, 0x42, 0x3C, 0x00},
	'H': {0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00},
	'I': {0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00},
	'J': {0x07, 0x02, 0x02, 0x02, 0x02, 0x42, 0x3C, 0x00},
	'K': {0x42, 0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x00},
	'L': {0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00},
	'M': {0x81, 0xC3, 0xA5, 0x99, 0x81, 0x81, 0x81, 0x00},
	'N': {0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x42, 0x00},
	'O': {0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00},
	'P': {0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x00},
	'Q': {0x3C, 0x42, 0x42, 0x42, 0x4A, 0x44, 0x3A, 0x00},
	'R': {0x7C, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x42, 0x00},
	'S': {0x3C, 0x42, 0x40, 0x3C, 0x02, 0x42, 0x3C, 0x00},
	'T': {0x7E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00},
	'U': {0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00},
	'V': {0x42, 0x42, 0x42, 0x42, 0x24, 0x24, 0x18, 0x00},
	'W': {0x81, 0x81, 0x81, 0x99, 0xA5, 0xC3, 0x81, 0x00},
	'X': {0x42, 0x24, 0x18, 0x10, 0x18, 0x24, 0x42, 0x00},
	'Y': {0x42, 0x42, 0x24, 0x18, 0x10, 0x10, 0x10, 0x00},
	'Z': {0x7E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x7E, 0x00},
	'a': {0x00, 0x00, 0x7C, 0x02, 0x7E, 0x44, 0x4A, 0x3E},
	'b': {0x40, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x00},
	'c': {0x00, 0x00, 0x3C, 0x40, 0x40, 0x42, 0x3C, 0x00},
	'd': {0x02, 0x02, 0x3E, 0x42, 0x42, 0x42, 0x3E, 0x00},
	'e': {0x00, 0x00, 0x3C, 0x42, 0x7E, 0x40, 0x3C, 0x00},
	'f': {0x1C, 0x22, 0x20, 0x7C, 0x20, 0x20, 0x20, 0x00},
	'g': {0x00, 0x00, 0x3E, 0x42, 0x42, 0x3E, 0x02, 0x3C},
	'h': {0x40, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x00},
	'i': {0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x1C, 0x00},
	'j': {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x44, 0x38},
	'k': {0x40, 0x40, 0x44, 0x48, 0x70, 0x48, 0x44, 0x00},
	'l': {0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00},
	'm': {0x00, 0x00, 0xEC, 0x92, 0x92, 0x92, 0x92, 0x00},
	'n': {0x00, 0x00, 0x74, 0x4A, 0x42, 0x42, 0x42, 0x00},
	'o': {0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x3C, 0x00},
	'p': {0x00, 0x00, 0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40},
	'q': {0x00, 0x00, 0x3E, 0x42, 0x42, 0x3E, 0x02, 0x02},
	'r': {0x00, 0x00, 0x5C, 0x62, 0x40, 0x40, 0x40, 0x00},
	's': {0x00, 0x00, 0x3E, 0x40, 0x3E, 0x02, 0x3E, 0x00},
	't': {0x20, 0x20, 0x7C, 0x20, 0x20, 0x22, 0x1C, 0x00},
	'u': {0x00, 0x00, 0x42, 0x42, 0x42, 0x46, 0x3A, 0x00},
	'v': {0x00, 0x00, 0x42, 0x42, 0x24, 0x24, 0x18, 0x00},
	'w': {0x00, 0x00, 0x81, 0x99, 0xA5, 0x5A, 0x42, 0x00},
	'x': {0x00, 0x00, 0x42, 0x24, 0x18, 0x24, 0x42, 0x00},
	'y': {0x00, 0x00, 0x42, 0x42, 0x3E, 0x02, 0x02, 0x3C},
	'z': {0x00, 0x00, 0x7E, 0x08, 0x10, 0x20, 0x7E, 0x00},
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
	'/': {0x00, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00},
	'!': {0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x08, 0x00},
	'-': {0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00},
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
	Cooking      bool   `json:"cooking"`
	Fire         bool   `json:"fire"`
	Smoking      bool   `json:"smoking"`
	Sleeping     bool   `json:"sleeping"`
	SOPViolation bool   `json:"sop_violation"`
	Description  string `json:"description"`
}

// OpenAI API request / response structs
type OpenAIRequest struct {
	Model          string          `json:"model"`
	Messages       []OpenAIMessage `json:"messages"`
	ResponseFormat *OpenAIRespFmt  `json:"response_format,omitempty"`
}

type OpenAIMessage struct {
	Role    string          `json:"role"`
	Content []OpenAIContent `json:"content"`
}

type OpenAIContent struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *OpenAIImageURL `json:"image_url,omitempty"`
}

type OpenAIImageURL struct {
	URL string `json:"url"`
}

type OpenAIRespFmt struct {
	Type string `json:"type"`
}

type OpenAIResponse struct {
	Choices []OpenAIChoice `json:"choices"`
}

type OpenAIChoice struct {
	Message OpenAIMsgContent `json:"message"`
}

type OpenAIMsgContent struct {
	Content string `json:"content"`
}

// callVLMAPI mengoordinasikan pemanggilan VLM secara asinkron berdasarkan provider aktif (Gemini atau OpenAI)
func (sp *StreamProcessor) callVLMAPI(jpegData []byte) {
	sp.geminiBusyMu.Lock()
	sp.geminiBusy = true
	sp.geminiBusyMu.Unlock()

	go func() {
		defer func() {
			sp.geminiBusyMu.Lock()
			sp.geminiBusy = false
			sp.geminiBusyMu.Unlock()
		}()

		if sp.vlmProvider == "openai" {
			sp.executeOpenAIAPI(jpegData)
		} else {
			sp.executeGeminiAPI(jpegData)
		}
	}()
}

// executeGeminiAPI mengirimkan frame JPEG ke Google Gemini API secara sinkron (di dalam goroutine)
func (sp *StreamProcessor) executeGeminiAPI(jpegData []byte) {
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

	jsonBytes, err := json.Marshal(reqPayload)
	if err != nil {
		log.Printf("Gagal marshal request payload Gemini: %v", err)
		return
	}

	// URL endpoint Gemini API (menggunakan model gemini-2.5-flash)
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=%s", sp.geminiAPIKey)

	// Buat HTTP Request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Printf("Gagal membuat HTTP request Gemini: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-goog-api-key", sp.geminiAPIKey)

	// Eksekusi HTTP Request
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Gagal menghubungi Gemini API: %v", err)
		sp.statusMu.Lock()
		sp.geminiDescription = fmt.Sprintf("Error: Gagal menghubungi Gemini API (%v)", err)
		sp.statusMu.Unlock()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("Gemini API mengembalikan status error: %d, Response: %s", resp.StatusCode, string(bodyBytes))
		sp.statusMu.Lock()
		sp.geminiDescription = fmt.Sprintf("Error: Gemini API mengembalikan kode %d (Periksa Gemini API Key)", resp.StatusCode)
		sp.statusMu.Unlock()
		return
	}

	// Decode response
	var geminiResp GeminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		log.Printf("Gagal decode response Gemini API: %v", err)
		return
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		log.Printf("Gemini API tidak mengembalikan konten candidates.")
		sp.statusMu.Lock()
		sp.geminiDescription = "Error: Gemini mengembalikan hasil kosong (Candidates Empty)"
		sp.statusMu.Unlock()
		return
	}

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
	
	// Jika terjadi transisi alarm kebakaran
	fireTriggered := result.Fire && !sp.geminiFireAlert
	sp.geminiFireAlert = result.Fire
 
	// Filter 2x berurutan untuk rokok, tidur, dan SOP pelanggaran agar tidak terlalu sensitif/spam
	if result.Smoking {
		sp.geminiSmokingCount++
	} else {
		sp.geminiSmokingCount = 0
	}
	smokingTriggered := false
	if sp.geminiSmokingCount >= 2 && !sp.geminiSmokingAlert {
		smokingTriggered = true
		sp.geminiSmokingAlert = true
	} else if sp.geminiSmokingCount == 0 && sp.geminiSmokingAlert {
		sp.geminiSmokingAlert = false
	}
 
	if result.Sleeping {
		sp.geminiSleepingCount++
	} else {
		sp.geminiSleepingCount = 0
	}
	sleepingTriggered := false
	if sp.geminiSleepingCount >= 2 && !sp.geminiSleepingAlert {
		sleepingTriggered = true
		sp.geminiSleepingAlert = true
	} else if sp.geminiSleepingCount == 0 && sp.geminiSleepingAlert {
		sp.geminiSleepingAlert = false
	}
 
	if result.SOPViolation {
		sp.geminiSOPViolationCount++
	} else {
		sp.geminiSOPViolationCount = 0
	}
	sopViolationTriggered := false
	if sp.geminiSOPViolationCount >= 2 && !sp.geminiSOPViolationAlert {
		sopViolationTriggered = true
		sp.geminiSOPViolationAlert = true
	} else if sp.geminiSOPViolationCount == 0 && sp.geminiSOPViolationAlert {
		sp.geminiSOPViolationAlert = false
	}
	sp.statusMu.Unlock()

	// Tangani alarm kebakaran aktif
	if fireTriggered {
		log.Printf("WARNING!!! DETEKSI KEBAKARAN/API DARI GEMINI: %s", result.Description)
		// Simpan bukti foto snapshot kebakaran
		go sp.saveFireSnapshot(jpegData, nil)
		// Rekam otomatis foto bersih & koordinat kotak pembatas dari Guru VLM untuk melatih Murid YOLO
		go sp.recordAlarmToTrainingDataset(jpegData)
	}

	// Tangani alarm merokok aktif
	if smokingTriggered {
		log.Printf("WARNING!!! DETEKSI MEROKOK DARI GEMINI: %s", result.Description)
		go sp.saveEventSnapshot(jpegData, "smoking")
		// Rekam otomatis foto bersih & koordinat kotak pembatas dari Guru VLM untuk melatih Murid YOLO
		go sp.recordAlarmToTrainingDataset(jpegData)
	}

	// Tangani alarm tidur aktif
	if sleepingTriggered {
		log.Printf("WARNING!!! DETEKSI TIDUR DARI GEMINI: %s", result.Description)
		go sp.saveEventSnapshot(jpegData, "sleeping")
		// Rekam otomatis foto bersih & koordinat kotak pembatas dari Guru VLM untuk melatih Murid YOLO
		go sp.recordAlarmToTrainingDataset(jpegData)
	}

	// Tangani alarm pelanggaran SOP/K3 aktif
	if sopViolationTriggered {
		log.Printf("WARNING!!! DETEKSI PELANGGARAN SOP/K3 DARI GEMINI: %s", result.Description)
		go sp.saveEventSnapshot(jpegData, "sop_violation")
		go sp.recordAlarmToTrainingDataset(jpegData)
	}
}

// executeOpenAIAPI mengirimkan frame JPEG ke OpenAI Chat Completion API secara sinkron (di dalam goroutine)
func (sp *StreamProcessor) executeOpenAIAPI(jpegData []byte) {
	// Encode frame JPEG ke Base64
	base64Data := base64.StdEncoding.EncodeToString(jpegData)

	// Payload request sesuai format OpenAI Chat Completion dengan gambar
	reqPayload := OpenAIRequest{
		Model: "gpt-4o-mini", // Model tercepat dan termurah untuk analisis visi
		Messages: []OpenAIMessage{
			{
				Role: "user",
				Content: []OpenAIContent{
					{
						Type: "text",
						Text: sp.geminiPrompt,
					},
					{
						Type: "image_url",
						ImageURL: &OpenAIImageURL{
							URL: fmt.Sprintf("data:image/jpeg;base64,%s", base64Data),
						},
					},
				},
			},
		},
		ResponseFormat: &OpenAIRespFmt{
			Type: "json_object",
		},
	}

	// Marshal ke JSON
	jsonBytes, err := json.Marshal(reqPayload)
	if err != nil {
		log.Printf("Gagal marshal request payload OpenAI: %v", err)
		return
	}

	// URL endpoint OpenAI API
	url := "https://api.openai.com/v1/chat/completions"

	// Buat HTTP Request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Printf("Gagal membuat HTTP request OpenAI: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", sp.openaiAPIKey))

	// Eksekusi HTTP Request
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Gagal menghubungi OpenAI API: %v", err)
		sp.statusMu.Lock()
		sp.geminiDescription = fmt.Sprintf("Error: Gagal menghubungi OpenAI API (%v)", err)
		sp.statusMu.Unlock()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("OpenAI API mengembalikan status error: %d, Response: %s", resp.StatusCode, string(bodyBytes))
		sp.statusMu.Lock()
		sp.geminiDescription = fmt.Sprintf("Error: OpenAI API mengembalikan kode %d (Periksa OpenAI API Key)", resp.StatusCode)
		sp.statusMu.Unlock()
		return
	}

	// Decode Response
	var openAIResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		log.Printf("Gagal decode response OpenAI API: %v", err)
		return
	}

	if len(openAIResp.Choices) == 0 {
		log.Printf("OpenAI API tidak mengembalikan konten choices.")
		sp.statusMu.Lock()
		sp.geminiDescription = "Error: OpenAI mengembalikan hasil kosong (Choices Empty)"
		sp.statusMu.Unlock()
		return
	}

	rawText := openAIResp.Choices[0].Message.Content
	cleanJSON := sanitizeJSON(rawText)

	var result GeminiAnalysisResult
	if err := json.Unmarshal([]byte(cleanJSON), &result); err != nil {
		log.Printf("Gagal unmarshal hasil teks OpenAI ke JSON: %v. Raw text: %s", err, rawText)
		sp.statusMu.Lock()
		sp.geminiDescription = fmt.Sprintf("Error: Gagal membaca format teks kognitif OpenAI (%v)", err)
		sp.statusMu.Unlock()
		return
	}

	// Simpan hasil ke state StreamProcessor secara thread-safe
	sp.statusMu.Lock()
	sp.geminiDescription = result.Description
	sp.geminiCookingAlert = result.Cooking
	
	// Jika terjadi transisi alarm kebakaran
	fireTriggered := result.Fire && !sp.geminiFireAlert
	sp.geminiFireAlert = result.Fire
 
	// Filter 2x berurutan untuk rokok, tidur, dan SOP pelanggaran agar tidak terlalu sensitif/spam
	if result.Smoking {
		sp.geminiSmokingCount++
	} else {
		sp.geminiSmokingCount = 0
	}
	smokingTriggered := false
	if sp.geminiSmokingCount >= 2 && !sp.geminiSmokingAlert {
		smokingTriggered = true
		sp.geminiSmokingAlert = true
	} else if sp.geminiSmokingCount == 0 && sp.geminiSmokingAlert {
		sp.geminiSmokingAlert = false
	}
 
	if result.Sleeping {
		sp.geminiSleepingCount++
	} else {
		sp.geminiSleepingCount = 0
	}
	sleepingTriggered := false
	if sp.geminiSleepingCount >= 2 && !sp.geminiSleepingAlert {
		sleepingTriggered = true
		sp.geminiSleepingAlert = true
	} else if sp.geminiSleepingCount == 0 && sp.geminiSleepingAlert {
		sp.geminiSleepingAlert = false
	}
 
	if result.SOPViolation {
		sp.geminiSOPViolationCount++
	} else {
		sp.geminiSOPViolationCount = 0
	}
	sopViolationTriggered := false
	if sp.geminiSOPViolationCount >= 2 && !sp.geminiSOPViolationAlert {
		sopViolationTriggered = true
		sp.geminiSOPViolationAlert = true
	} else if sp.geminiSOPViolationCount == 0 && sp.geminiSOPViolationAlert {
		sp.geminiSOPViolationAlert = false
	}
	sp.statusMu.Unlock()

	// Tangani alarm kebakaran aktif
	if fireTriggered {
		log.Printf("WARNING!!! DETEKSI KEBAKARAN/API DARI OPENAI: %s", result.Description)
		go sp.saveFireSnapshot(jpegData, nil)
		// Rekam otomatis foto bersih & koordinat kotak pembatas dari Guru VLM untuk melatih Murid YOLO
		go sp.recordAlarmToTrainingDataset(jpegData)
	}

	// Tangani alarm merokok aktif
	if smokingTriggered {
		log.Printf("WARNING!!! DETEKSI MEROKOK DARI OPENAI: %s", result.Description)
		go sp.saveEventSnapshot(jpegData, "smoking")
		// Rekam otomatis foto bersih & koordinat kotak pembatas dari Guru VLM untuk melatih Murid YOLO
		go sp.recordAlarmToTrainingDataset(jpegData)
	}

	// Tangani alarm tidur aktif
	if sleepingTriggered {
		log.Printf("WARNING!!! DETEKSI TIDUR DARI OPENAI: %s", result.Description)
		go sp.saveEventSnapshot(jpegData, "sleeping")
		// Rekam otomatis foto bersih & koordinat kotak pembatas dari Guru VLM untuk melatih Murid YOLO
		go sp.recordAlarmToTrainingDataset(jpegData)
	}

	// Tangani alarm pelanggaran SOP/K3 aktif
	if sopViolationTriggered {
		log.Printf("WARNING!!! DETEKSI PELANGGARAN SOP/K3 DARI OPENAI: %s", result.Description)
		go sp.saveEventSnapshot(jpegData, "sop_violation")
		go sp.recordAlarmToTrainingDataset(jpegData)
	}
}

// sanitizeJSON membersihkan format JSON mentah dari blok tag markdown kustom (seperti ```json ... ```)
func sanitizeJSON(input string) string {
	startIdx := -1
	endIdx := -1
	
	// Cari karakter kurung buka pertama ({ atau [)
	for i, r := range input {
		if r == '{' || r == '[' {
			startIdx = i
			break
		}
	}
	
	// Cari karakter kurung tutup terakhir (} atau ])
	for i := len(input) - 1; i >= 0; i-- {
		if input[i] == '}' || input[i] == ']' {
			endIdx = i
			break
		}
	}
	
	if startIdx == -1 || endIdx == -1 || startIdx >= endIdx {
		return input
	}
	return input[startIdx : endIdx+1]
}

// parseDetectionsJSON menguraikan teks JSON deteksi (baik berwujud array langsung maupun terbungkus objek)
// menjadi slice []GeminiAutoLabelResult secara toleran dan fleksibel.
func parseDetectionsJSON(cleanJSON string) ([]GeminiAutoLabelResult, error) {
	cleanJSON = strings.TrimSpace(cleanJSON)
	if cleanJSON == "" {
		return nil, fmt.Errorf("empty JSON string")
	}

	// 1. Coba unmarshal langsung sebagai array []GeminiAutoLabelResult
	var list []GeminiAutoLabelResult
	if err := json.Unmarshal([]byte(cleanJSON), &list); err == nil {
		return list, nil
	}

	// 2. Coba unmarshal sebagai map objek, barangkali dibungkus key kustom oleh VLM
	var objMap map[string]json.RawMessage
	if err := json.Unmarshal([]byte(cleanJSON), &objMap); err == nil {
		// Cari key potensial yang berisi array deteksi
		for _, key := range []string{"detections", "objects", "boxes", "results", "data", "predictions"} {
			if raw, ok := objMap[key]; ok {
				var subList []GeminiAutoLabelResult
				if err := json.Unmarshal(raw, &subList); err == nil {
					return subList, nil
				}
			}
		}
	}

	// 3. Coba unmarshal sebagai objek tunggal GeminiAutoLabelResult (jika hanya ada 1 deteksi tanpa array)
	var single GeminiAutoLabelResult
	if err := json.Unmarshal([]byte(cleanJSON), &single); err == nil {
		return []GeminiAutoLabelResult{single}, nil
	}

	return nil, fmt.Errorf("failed to decode JSON into detection slice (raw text: %s)", cleanJSON)
}

// saveFireSnapshot menyimpan snapshot bukti kebakaran ke disk & log ke database SQLite
func (sp *StreamProcessor) saveFireSnapshot(jpegData []byte, detections []AIDetection) {
	// Gambar box terlebih dahulu sebelum disimpan (hanya sekali, jadi tidak membebani CPU)
	processedJPEG, _, _ := sp.drawBoundingBoxes(jpegData, detections)

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
		// Simpan otomatis foto bersih dan anotasi koordinat YOLO ke dataset latihan jika ada deteksi lokal
		if len(detections) > 0 {
			go sp.saveToTrainingDataset(jpegData, detections)
		}
	}
}

// saveEventSnapshot menyimpan bukti foto kejadian kustom (seperti merokok atau tidur) ke disk & log ke SQLite
func (sp *StreamProcessor) saveEventSnapshot(jpegData []byte, eventType string) {
	// Tarik deteksi terakhir YOLO secara aman
	sp.detectionsMu.Lock()
	var detections []AIDetection
	if sp.lastDetections != nil {
		detections = make([]AIDetection, len(sp.lastDetections))
		copy(detections, sp.lastDetections)
	}
	sp.detectionsMu.Unlock()

	// Gambar box terlebih dahulu sebelum disimpan (hanya sekali, jadi tidak membebani CPU)
	processedJPEG, _, _ := sp.drawBoundingBoxes(jpegData, detections)

	// Nama file unik dengan timestamp
	snapshotFilename := fmt.Sprintf("%s_%s.jpg", eventType, time.Now().Format("20060102_150405_000"))
	snapshotPath := filepath.Join("snapshots", snapshotFilename)

	// Pastikan folder snapshots ada
	os.MkdirAll("snapshots", 0755)

	err := os.WriteFile(snapshotPath, processedJPEG, 0644)
	if err != nil {
		log.Printf("Gagal menyimpan file snapshot %s: %v", eventType, err)
		return
	}

	// Catat alarm kejadian ke database SQLite
	_, dbErr := sp.db.LogDetection(eventType, 0.95, snapshotFilename) // Default confidence 95%
	if dbErr != nil {
		log.Printf("Gagal mencatat log %s ke SQLite: %v", eventType, dbErr)
	} else {
		log.Printf("DARURAT %s TERCATAT! Foto disimpan: %s", strings.ToUpper(eventType), snapshotFilename)
	}
}

// saveToTrainingDataset menyimpan foto bersih dan anotasi label YOLO ke folder dataset secara otomatis
func (sp *StreamProcessor) saveToTrainingDataset(cleanJPEG []byte, detections []AIDetection) {
	if len(detections) == 0 {
		return
	}

	// Buat folder dataset tujuan jika belum ada
	imgDir := filepath.Join("dataset", "images", "train")
	lblDir := filepath.Join("dataset", "labels", "train")
	os.MkdirAll(imgDir, 0755)
	os.MkdirAll(lblDir, 0755)

	// Nama file unik dengan timestamp
	timestamp := time.Now().Format("20060102_150405")
	baseName := fmt.Sprintf("auto_cctv_%s_%d", timestamp, time.Now().UnixNano()%1000)
	
	imgPath := filepath.Join(imgDir, baseName+".jpg")
	lblPath := filepath.Join(lblDir, baseName+".txt")

	// Decode dimensi gambar asli untuk koordinat normalisasi YOLO
	imgConfig, _, errConfig := image.DecodeConfig(bytes.NewReader(cleanJPEG))
	if errConfig != nil {
		log.Printf("[Auto-Dataset] Gagal membaca dimensi gambar asli: %v", errConfig)
		return
	}
	width := float64(imgConfig.Width)
	height := float64(imgConfig.Height)

	if width <= 0 || height <= 0 {
		log.Printf("[Auto-Dataset] Dimensi gambar tidak valid: %f x %f", width, height)
		return
	}

	var txtLines []string
	for _, det := range detections {
		if len(det.Box) < 4 {
			continue
		}

		// Saring berdasarkan threshold kepercayaan
		if det.Confidence < sp.confThreshold {
			continue
		}

		// Konversi koordinat absolut piksel ke koordinat pusat + rasio YOLO (0.0 s.d 1.0)
		x1, y1 := det.Box[0], det.Box[1]
		x2, y2 := det.Box[2], det.Box[3]

		// Batasi ke dimensi gambar
		if x1 < 0 { x1 = 0 }
		if y1 < 0 { y1 = 0 }
		if x2 > width { x2 = width }
		if y2 > height { y2 = height }

		wBox := x2 - x1
		hBox := y2 - y1

		xCenter := (x1 + x2) / 2 / width
		yCenter := (y1 + y2) / 2 / height
		wRatio := wBox / width
		hRatio := hBox / height

		// Petakan nama label ke Class ID dataset
		labelTrimmed := strings.TrimSpace(det.Label)
		if labelTrimmed == "person" {
			labelTrimmed = "manusia"
		} else if labelTrimmed == "fire" {
			labelTrimmed = "api"
		} else if labelTrimmed == "smoke" {
			labelTrimmed = "asap"
		}

		classID, err := getOrAddClassID(labelTrimmed)
		if err != nil {
			log.Printf("[Auto-Dataset] Gagal memetakan kelas '%s': %v", labelTrimmed, err)
			continue
		}

		line := fmt.Sprintf("%d %.6f %.6f %.6f %.6f", classID, xCenter, yCenter, wRatio, hRatio)
		txtLines = append(txtLines, line)
	}

	// Jika tidak ada deteksi yang valid setelah disaring threshold, tidak perlu disimpan
	if len(txtLines) == 0 {
		return
	}

	// Simpan gambar bersih
	err := os.WriteFile(imgPath, cleanJPEG, 0644)
	if err != nil {
		log.Printf("[Auto-Dataset] Gagal menyimpan file gambar latihan: %v", err)
		return
	}

	// Simpan file label koordinat YOLO
	txtContent := strings.Join(txtLines, "\n") + "\n"
	err = os.WriteFile(lblPath, []byte(txtContent), 0644)
	if err != nil {
		log.Printf("[Auto-Dataset] Gagal menyimpan file label latihan: %v", err)
		os.Remove(imgPath) // hapus gambar jika teks gagal disimpan
		return
	}

	log.Printf("[Auto-Dataset] DATASET BERTAMBAH OTOMATIS! Gambar & label tersimpan: %s", baseName)

	// Cek apakah data baru bertambah kelipatan 10 untuk memicu training otomatis secara asinkron
	go checkAndTriggerAutoTraining()
}

// recordAlarmToTrainingDataset secara asinkron memanggil auto-label VLM untuk mendeteksi koordinat objek pada frame alarm,
// lalu menyimpannya langsung ke dataset pelatihan YOLO agar murid YOLO bisa belajar dari guru VLM.
func (sp *StreamProcessor) recordAlarmToTrainingDataset(jpegData []byte) {
	// Dapatkan API Key dan Provider secara thread-safe
	sp.statusMu.Lock()
	vlmProvider := sp.vlmProvider
	openaiAPIKey := sp.openaiAPIKey
	geminiAPIKey := sp.geminiAPIKey
	sp.statusMu.Unlock()

	var detections []AIDetection
	var err error

	// Panggil VLM untuk melokalisasi koordinat kotak pembatas (auto-labeling)
	if vlmProvider == "openai" && openaiAPIKey != "" {
		detections, err = detectObjectsWithOpenAI(jpegData, openaiAPIKey)
	} else if geminiAPIKey != "" {
		detections, err = detectObjectsWithGemini(jpegData, geminiAPIKey)
	}

	if err != nil {
		log.Printf("[Auto-Learning] Gagal auto-labeling untuk perekaman otomatis dataset alarm: %v", err)
		return
	}

	if len(detections) > 0 {
		// Simpan gambar bersih dan file label .txt standar YOLO ke folder latihan
		sp.saveToTrainingDataset(jpegData, detections)
		log.Printf("[Auto-Learning] Pembelajaran Mandiri: Berhasil merekam %d objek dari Guru VLM ke dataset training YOLO!", len(detections))
	}
}


