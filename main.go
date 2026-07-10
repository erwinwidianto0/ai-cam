package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// Config menyimpan pengaturan aplikasi
type Config struct {
	RTSPURL            string  `json:"rtsp_url"`
	AIEndpoint         string  `json:"ai_endpoint"`
	ConfThreshold      float64 `json:"conf_threshold"`
	Port               string  `json:"port"`
	ZoneXMin           float64 `json:"zone_x_min"`
	ZoneYMin           float64 `json:"zone_y_min"`
	ZoneXMax           float64 `json:"zone_x_max"`
	ZoneYMax           float64 `json:"zone_y_max"`
	CookingTriggerSecs int     `json:"cooking_trigger_secs"`
}

var (
	config     Config
	configMu   sync.RWMutex
	configPath = "config.json"

	dbManager       *DBManager
	streamProcessor *StreamProcessor
	processorMu     sync.Mutex
)

// loadConfig membaca file konfigurasi atau membuat konfigurasi default
func loadConfig() error {
	configMu.Lock()
	defer configMu.Unlock()

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// Konfigurasi default (menggunakan CCTV RTSP Anda)
		config = Config{
			RTSPURL:            "rtsp://admin:%40BANDUNGhijau2023@192.168.1.59:554/h264Preview_01_main",
			AIEndpoint:         "http://127.0.0.1:8000",
			ConfThreshold:      0.55, // Batas kepercayaan 55%
			Port:               "8080",
			ZoneXMin:           0.30,
			ZoneYMin:           0.20,
			ZoneXMax:           0.70,
			ZoneYMax:           0.90,
			CookingTriggerSecs: 8, // 8 detik berada di kompor untuk memicu deteksi memasak
		}
		
		file, err := json.MarshalIndent(config, "", "  ")
		if err != nil {
			return err
		}
		return os.WriteFile(configPath, file, 0644)
	}

	file, err := os.ReadFile(configPath)
	if err != nil {
		return err
	}
	err = json.Unmarshal(file, &config)
	if err != nil {
		return err
	}

	// Tambahkan fallback nilai default jika variabel baru belum ada di config.json
	if config.ZoneXMax == 0 && config.ZoneYMax == 0 {
		config.ZoneXMin = 0.30
		config.ZoneYMin = 0.20
		config.ZoneXMax = 0.70
		config.ZoneYMax = 0.90
		config.CookingTriggerSecs = 8
		
		// Simpan perubahan secara diam-diam agar config.json ter-update
		fileOut, _ := json.MarshalIndent(config, "", "  ")
		os.WriteFile(configPath, fileOut, 0644)
	}

	return nil
}

// saveConfig menyimpan konfigurasi terkini ke file
func saveConfig(cfg Config) error {
	configMu.Lock()
	config = cfg
	configMu.Unlock()

	file, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(configPath, file, 0644)
}

func main() {
	log.Println("Memulai aplikasi CCTV AI-CAM...")

	// 1. Muat Konfigurasi
	if err := loadConfig(); err != nil {
		log.Fatalf("Gagal memuat konfigurasi: %v", err)
	}
	log.Printf("Konfigurasi dimuat. Menjalankan di port %s", config.Port)

	// Buat folder snapshots
	os.MkdirAll("snapshots", 0755)

	// 2. Inisialisasi Database SQLite
	var err error
	dbManager, err = NewDBManager("aicam.db")
	if err != nil {
		log.Fatalf("Gagal menginisialisasi database: %v", err)
	}
	defer dbManager.Close()
	log.Println("Database SQLite diinisialisasi sukses.")

	// 3. Mulai Processor CCTV
	aiClient := NewAIClient(config.AIEndpoint)
	streamProcessor = NewStreamProcessor(
		config.RTSPURL, 
		aiClient, 
		dbManager, 
		config.ConfThreshold,
		config.ZoneXMin,
		config.ZoneYMin,
		config.ZoneXMax,
		config.ZoneYMax,
		config.CookingTriggerSecs,
	)
	streamProcessor.Start()
	defer streamProcessor.Stop()

	// 4. Daftarkan HTTP Routes
	// Melayani file statik untuk dashboard
	fs := http.FileServer(http.Dir("./static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	http.HandleFunc("/", handleHome)

	// Endpoint streaming video
	http.HandleFunc("/stream", handleStream)

	// API Endpoints
	http.HandleFunc("/api/events", handleAPIEvents)
	http.HandleFunc("/api/stats", handleAPIStats)
	http.HandleFunc("/api/settings", handleAPISettings)

	// Endpoint menyajikan file gambar snapshot
	http.Handle("/snapshots/", http.StripPrefix("/snapshots/", http.FileServer(http.Dir("./snapshots"))))

	// Jalankan server
	addr := ":" + config.Port
	log.Printf("Server dashboard aktif di http://localhost%s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Gagal menjalankan server web: %v", err)
	}
}

// handleHome melayani file index.html pada root path
func handleHome(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "./static/index.html")
}

// handleStream menyalurkan live stream video MJPEG ke browser
func handleStream(w http.ResponseWriter, r *http.Request) {
	frameChan := make(chan []byte, 10)
	
	processorMu.Lock()
	streamProcessor.AddClient(frameChan)
	processorMu.Unlock()

	defer func() {
		processorMu.Lock()
		streamProcessor.RemoveClient(frameChan)
		processorMu.Unlock()
		close(frameChan)
	}()

	// Atur header untuk streaming Multipart MJPEG
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	w.Header().Set("Cache-Control", "no-cache, private, no-store, must-revalidate, max-age=0")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Pragma", "no-cache")

	// Pastikan client menutup koneksi dengan benar
	notify := r.Context().Done()

	for {
		select {
		case frame, ok := <-frameChan:
			if !ok {
				return
			}
			// Format header per frame untuk standard MJPEG
			_, err := fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(frame))
			if err != nil {
				return
			}
			_, err = w.Write(frame)
			if err != nil {
				return
			}
			_, err = w.Write([]byte("\r\n"))
			if err != nil {
				return
			}
			
			// Flush buffer agar frame terkirim seketika
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			
		case <-notify:
			return
		}
	}
}

// handleAPIEvents menyajikan data riwayat deteksi dalam format JSON
func handleAPIEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	limitStr := r.URL.Query().Get("limit")
	limit := 15 // Default limit 15 baris data
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	detections, err := dbManager.GetRecentDetections(limit)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error": "%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(detections)
}

// handleAPIStats menyajikan data statistik real-time CCTV dan status AI
func handleAPIStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	dbStats, err := dbManager.GetStats()
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error": "%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	processorMu.Lock()
	fps := streamProcessor.fps
	latency := streamProcessor.aiLatency.Milliseconds()
	aiStatus := streamProcessor.aiStatus
	clientsCount := len(streamProcessor.clients)
	kitchenStatus := streamProcessor.kitchenStatus
	secondsInZone := streamProcessor.secondsInZone
	processorMu.Unlock()

	response := map[string]interface{}{
		"total_detections":  dbStats.TotalDetections,
		"detections_today":  dbStats.DetectionsToday,
		"stream_fps":        mathRound(fps, 1),
		"ai_latency_ms":     latency,
		"ai_status":         aiStatus,
		"active_viewers":    clientsCount,
		"kitchen_status":    kitchenStatus,
		"seconds_in_zone":   secondsInZone,
		"timestamp":         time.Now().Format("15:04:05"),
	}

	json.NewEncoder(w).Encode(response)
}

// mathRound membantu membulatkan nilai float ke presisi yang diinginkan
func mathRound(val float64, precision int) float64 {
	format := fmt.Sprintf("%%.%df", precision)
	resStr := fmt.Sprintf(format, val)
	res, _ := strconv.ParseFloat(resStr, 64)
	return res
}

// handleAPISettings membaca dan memperbarui konfigurasi runtime
func handleAPISettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method == http.MethodGet {
		configMu.RLock()
		defer configMu.RUnlock()
		json.NewEncoder(w).Encode(config)
		return
	}

	if r.Method == http.MethodPost {
		var newCfg Config
		err := json.NewDecoder(r.Body).Decode(&newCfg)
		if err != nil {
			http.Error(w, `{"error": "Invalid request body"}`, http.StatusBadRequest)
			return
		}

		if newCfg.RTSPURL == "" || newCfg.AIEndpoint == "" {
			http.Error(w, `{"error": "RTSP URL and AI Endpoint cannot be empty"}`, http.StatusBadRequest)
			return
		}

		if newCfg.ConfThreshold <= 0 || newCfg.ConfThreshold > 1 {
			newCfg.ConfThreshold = 0.55
		}

		configMu.RLock()
		oldRTSP := config.RTSPURL
		oldAI := config.AIEndpoint
		oldThreshold := config.ConfThreshold
		oldZXMin := config.ZoneXMin
		oldZYMin := config.ZoneYMin
		oldZXMax := config.ZoneXMax
		oldZYMax := config.ZoneYMax
		oldTriggerSecs := config.CookingTriggerSecs
		currentPort := config.Port // Ambil port aktif
		configMu.RUnlock()

		newCfg.Port = currentPort // Kembalikan port agar tidak kosong

		// Simpan konfigurasi baru
		if err := saveConfig(newCfg); err != nil {
			http.Error(w, fmt.Sprintf(`{"error": "Failed to save config: %s"}`, err.Error()), http.StatusInternalServerError)
			return
		}

		// Jika ada perubahan parameter apapun, restart streaming processor secara dinamis
		if oldRTSP != newCfg.RTSPURL || 
			oldAI != newCfg.AIEndpoint || 
			oldThreshold != newCfg.ConfThreshold ||
			oldZXMin != newCfg.ZoneXMin ||
			oldZYMin != newCfg.ZoneYMin ||
			oldZXMax != newCfg.ZoneXMax ||
			oldZYMax != newCfg.ZoneYMax ||
			oldTriggerSecs != newCfg.CookingTriggerSecs {
			
			log.Println("Mendeteksi perubahan konfigurasi. Mengatur ulang aliran CCTV...")
			
			processorMu.Lock()
			streamProcessor.Stop()
			
			aiClient := NewAIClient(newCfg.AIEndpoint)
			streamProcessor = NewStreamProcessor(
				newCfg.RTSPURL, 
				aiClient, 
				dbManager, 
				newCfg.ConfThreshold,
				newCfg.ZoneXMin,
				newCfg.ZoneYMin,
				newCfg.ZoneXMax,
				newCfg.ZoneYMax,
				newCfg.CookingTriggerSecs,
			)
			streamProcessor.Start()
			processorMu.Unlock()
			
			log.Println("Aliran CCTV berhasil diatur ulang dengan konfigurasi baru.")
		}

		json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Settings updated"})
		return
	}

	http.Error(w, `{"error": "Method not allowed"}`, http.StatusMethodNotAllowed)
}
