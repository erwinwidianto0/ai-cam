package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
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
	VLMProvider        string  `json:"vlm_provider"` // "gemini" atau "openai"
	OpenAIAPIKey       string  `json:"openai_api_key"`
	GeminiAPIKey       string  `json:"gemini_api_key"`
	GeminiPrompt       string  `json:"gemini_prompt"`
}

var (
	config     Config
	configMu   sync.RWMutex
	configPath = "config.json"

	dbManager       *DBManager
	streamProcessor *StreamProcessor
	processorMu     sync.Mutex

	// Variabel global untuk memantau status pelatihan AI lokal
	trainingActive  bool
	trainingPercent int
	trainingCmd     *exec.Cmd
	trainingMu      sync.Mutex
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
			VLMProvider:        "gemini",
			OpenAIAPIKey:       "",
			GeminiAPIKey:       "",
			GeminiPrompt:       "Analisis gambar CCTV dapur ini. Deteksi secara akurat: 1) Apakah ada orang sedang memasak di depan kompor (cooking: true/false)? 2) Apakah ada indikasi kebakaran, api, asap, atau tanda-tanda awal potensi kebakaran seperti kompor menyala tanpa pengawasan, asap mulai membubung, atau benda mudah terbakar terlalu dekat dengan api (fire: true/false)? 3) Apakah ada orang sedang merokok (smoking: true/false)? 4) Apakah ada orang sedang tidur (sleeping: true/false)? Kembalikan hasil dalam format JSON terstruktur dengan key: 'cooking' (boolean), 'fire' (boolean), 'smoking' (boolean), 'sleeping' (boolean), dan 'description' (string penjelasan singkat kondisi kejadian dan peringatan dini kebakaran dalam bahasa Indonesia).",
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
	updated := false
	if config.ZoneXMax == 0 && config.ZoneYMax == 0 {
		config.ZoneXMin = 0.30
		config.ZoneYMin = 0.20
		config.ZoneXMax = 0.70
		config.ZoneYMax = 0.90
		config.CookingTriggerSecs = 8
		updated = true
	}
	if config.GeminiPrompt == "" {
		config.GeminiPrompt = "Analisis gambar CCTV dapur ini. Deteksi secara akurat: 1) Apakah ada orang sedang memasak di depan kompor (cooking: true/false)? 2) Apakah ada indikasi kebakaran, api, atau asap (fire: true/false)? Kembalikan hasil dalam format JSON terstruktur dengan key: 'cooking' (boolean), 'fire' (boolean), dan 'description' (string penjelasan singkat kondisi kejadian dalam bahasa Indonesia)."
		updated = true
	}

	if updated {
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

	// Buat folder snapshots dan dataset lokal
	os.MkdirAll("snapshots", 0755)
	os.MkdirAll("dataset/raw", 0755)
	os.MkdirAll("dataset/images/train", 0755)
	os.MkdirAll("dataset/labels/train", 0755)

	// Perbaiki otomatis format data.yaml jika classes.txt sudah ada
	classesPath := "dataset/classes.txt"
	if content, err := os.ReadFile(classesPath); err == nil {
		var classes []string
		lines := strings.Split(string(content), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" {
				classes = append(classes, line)
			}
		}
		if len(classes) > 0 {
			rebuildDataYAML(classes)
		}
	}

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
		config.VLMProvider,
		config.OpenAIAPIKey,
		config.GeminiAPIKey,
		config.GeminiPrompt,
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
	http.HandleFunc("/api/settings/test-gemini", handleAPITestGemini)
	http.HandleFunc("/api/settings/test-openai", handleAPITestOpenAI)
	http.HandleFunc("/api/vlm/trigger", handleAPIVLMTrigger)

	// API Pelabelan dan Pelatihan Lokal
	http.HandleFunc("/api/label/images", handleAPILabelImages)
	http.HandleFunc("/api/label/save", handleAPILabelSave)
	http.HandleFunc("/api/label/upload", handleAPILabelUpload)
	http.HandleFunc("/api/label/images-labeled", handleAPILabelImagesLabeled)
	http.HandleFunc("/api/label/autodetect", handleAPILabelAutoDetect)
	http.HandleFunc("/api/train/start", handleAPITrainStart)
	http.HandleFunc("/api/train/status", handleAPITrainStatus)
	http.HandleFunc("/api/train/stop", handleAPITrainStop)

	// Endpoint menyajikan file gambar snapshot dan dataset
	http.Handle("/snapshots/", http.StripPrefix("/snapshots/", http.FileServer(http.Dir("./snapshots"))))
	http.Handle("/dataset/", http.StripPrefix("/dataset/", http.FileServer(http.Dir("./dataset"))))

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

	// Kirim status header 200 OK secara eksplisit
	w.WriteHeader(http.StatusOK)

	// Flush header seketika agar browser menghentikan spinner loadingnya
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

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
	
	// Data baru dari integrasi Gemini VLM
	geminiFireAlert := streamProcessor.geminiFireAlert
	geminiCookingAlert := streamProcessor.geminiCookingAlert
	geminiSmokingAlert := streamProcessor.geminiSmokingAlert
	geminiSleepingAlert := streamProcessor.geminiSleepingAlert
	geminiDescription := streamProcessor.geminiDescription
	processorMu.Unlock()

	configMu.RLock()
	geminiActive := config.GeminiAPIKey != ""
	configMu.RUnlock()

	response := map[string]interface{}{
		"total_detections":     dbStats.TotalDetections,
		"detections_today":     dbStats.DetectionsToday,
		"stream_fps":           mathRound(fps, 1),
		"ai_latency_ms":        latency,
		"ai_status":            aiStatus,
		"active_viewers":       clientsCount,
		"kitchen_status":       kitchenStatus,
		"seconds_in_zone":      secondsInZone,
		"gemini_fire_alert":     geminiFireAlert,
		"gemini_cooking_alert":  geminiCookingAlert,
		"gemini_smoking_alert":  geminiSmokingAlert,
		"gemini_sleeping_alert": geminiSleepingAlert,
		"gemini_description":    geminiDescription,
		"gemini_active":        geminiActive,
		"timestamp":            time.Now().Format("15:04:05"),
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
		oldVLMProvider := config.VLMProvider
		oldOpenAIAPIKey := config.OpenAIAPIKey
		oldGeminiAPIKey := config.GeminiAPIKey
		oldGeminiPrompt := config.GeminiPrompt
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
			oldTriggerSecs != newCfg.CookingTriggerSecs ||
			oldVLMProvider != newCfg.VLMProvider ||
			oldOpenAIAPIKey != newCfg.OpenAIAPIKey ||
			oldGeminiAPIKey != newCfg.GeminiAPIKey ||
			oldGeminiPrompt != newCfg.GeminiPrompt {
			
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
				newCfg.VLMProvider,
				newCfg.OpenAIAPIKey,
				newCfg.GeminiAPIKey,
				newCfg.GeminiPrompt,
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

// handleAPITestGemini memvalidasi API Key Gemini ke Google secara langsung dengan query teks ringan
func handleAPITestGemini(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		http.Error(w, `{"status": "error", "message": "Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var reqBody struct {
		GeminiAPIKey string `json:"gemini_api_key"`
	}
	err := json.NewDecoder(r.Body).Decode(&reqBody)
	if err != nil || reqBody.GeminiAPIKey == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "API Key tidak boleh kosong"})
		return
	}

	// Payload request ringan hanya kirim teks pendek untuk verifikasi validitas key
	testPayload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]interface{}{
					{
						"text": "Hello, is this API Key valid? Reply with YES.",
					},
				},
			},
		},
	}

	jsonBytes, err := json.Marshal(testPayload)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal menyusun payload: %v", err)})
		return
	}

	// Panggil Gemini API Endpoint
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=%s", reqBody.GeminiAPIKey)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal inisialisasi request HTTP: %v", err)})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-goog-api-key", reqBody.GeminiAPIKey) // Header penting untuk token API bertipe tertentu

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		w.WriteHeader(http.StatusOK) // Kembalikan HTTP 200 agar ditangkap frontend secara ramah
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal terhubung ke Google API (%v)", err)})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "error",
			"message": fmt.Sprintf("Kunci API tidak valid (Status HTTP %d)", resp.StatusCode),
		})
		return
	}
	// Jika status 200 OK, berarti kunci API valid!
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Koneksi sukses! API Key Valid.",
	})
}

// ==========================================
// LOGIKA PELABELAN & PELATIHAN ASINKRON LOKAL
// ==========================================

func moveFile(src, dst string) error {
	err := os.Rename(src, dst)
	if err == nil {
		return nil
	}
	// Fallback ke copy dan delete
	input, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	err = os.WriteFile(dst, input, 0644)
	if err != nil {
		return err
	}
	return os.Remove(src)
}

func getOrAddClassID(label string) (int, error) {
	classesPath := "dataset/classes.txt"
	content, err := os.ReadFile(classesPath)
	var classes []string
	if err == nil {
		lines := strings.Split(string(content), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" {
				classes = append(classes, line)
			}
		}
	}

	// Cari indeks
	for idx, c := range classes {
		if strings.ToLower(c) == strings.ToLower(label) {
			return idx, nil
		}
	}

	// Tambahkan baru
	classes = append(classes, label)
	err = os.WriteFile(classesPath, []byte(strings.Join(classes, "\n")+"\n"), 0644)
	if err != nil {
		return 0, err
	}

	// Tulis ulang data.yaml
	err = rebuildDataYAML(classes)
	if err != nil {
		return 0, err
	}

	return len(classes) - 1, nil
}

func rebuildDataYAML(classes []string) error {
	yamlPath := "dataset/data.yaml"
	var formattedNames []string
	for _, c := range classes {
		formattedNames = append(formattedNames, fmt.Sprintf("'%s'", c))
	}
	yamlContent := fmt.Sprintf(`path: ./dataset
train: images/train
val: images/train
nc: %d
names: [%s]
`, len(classes), strings.Join(formattedNames, ", "))
	return os.WriteFile(yamlPath, []byte(yamlContent), 0644)
}

func findLatestBestPt() (string, error) {
	defaultPath := filepath.Join("runs", "detect", "train", "weights", "best.pt")
	if _, err := os.Stat(defaultPath); err == nil {
		return defaultPath, nil
	}

	detectDir := filepath.Join("runs", "detect")
	files, err := os.ReadDir(detectDir)
	if err != nil {
		return "", err
	}

	var latestDir string
	var latestTime time.Time

	for _, file := range files {
		if file.IsDir() && strings.HasPrefix(file.Name(), "train") {
			dirPath := filepath.Join(detectDir, file.Name())
			info, err := os.Stat(dirPath)
			if err == nil {
				if latestDir == "" || info.ModTime().After(latestTime) {
					latestDir = dirPath
					latestTime = info.ModTime()
				}
			}
		}
	}

	if latestDir == "" {
		return "", fmt.Errorf("tidak menemukan folder runs/detect/train*")
	}

	bestPt := filepath.Join(latestDir, "weights", "best.pt")
	if _, err := os.Stat(bestPt); err != nil {
		return "", fmt.Errorf("tidak menemukan file best.pt di %s", bestPt)
	}

	return bestPt, nil
}

func appendTrainingLog(line string) {
	logFile, err := os.OpenFile("snapshots/training.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		defer logFile.Close()
		logFile.WriteString(line + "\n")
	}
}

func handleAPILabelImages(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	files, err := os.ReadDir("dataset/raw")
	if err != nil {
		json.NewEncoder(w).Encode([]string{})
		return
	}

	var images []string
	for _, file := range files {
		if !file.IsDir() {
			name := file.Name()
			ext := strings.ToLower(filepath.Ext(name))
			if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
				images = append(images, name)
			}
		}
	}
	json.NewEncoder(w).Encode(images)
}

type LabelBox struct {
	Label string  `json:"label"`
	XMin  float64 `json:"x_min"`
	YMin  float64 `json:"y_min"`
	XMax  float64 `json:"x_max"`
	YMax  float64 `json:"y_max"`
}

type LabelSaveRequest struct {
	Filename string     `json:"filename"`
	Boxes    []LabelBox `json:"boxes"`
}

func handleAPILabelSave(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var req LabelSaveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Invalid JSON"})
		return
	}

	if req.Filename == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Filename is required"})
		return
	}

	ext := filepath.Ext(req.Filename)
	nameWithoutExt := strings.TrimSuffix(req.Filename, ext)
	txtPath := filepath.Join("dataset", "labels", "train", nameWithoutExt+".txt")

	var txtLines []string
	for _, box := range req.Boxes {
		labelTrimmed := strings.TrimSpace(box.Label)
		if labelTrimmed == "" {
			continue
		}

		classID, err := getOrAddClassID(labelTrimmed)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal memetakan kelas: %v", err)})
			return
		}

		xCenter := (box.XMin + box.XMax) / 2
		yCenter := (box.YMin + box.YMax) / 2
		width := box.XMax - box.XMin
		height := box.YMax - box.YMin

		if xCenter < 0 { xCenter = 0 } else if xCenter > 1 { xCenter = 1 }
		if yCenter < 0 { yCenter = 0 } else if yCenter > 1 { yCenter = 1 }
		if width < 0 { width = 0 } else if width > 1 { width = 1 }
		if height < 0 { height = 0 } else if height > 1 { height = 1 }

		line := fmt.Sprintf("%d %.6f %.6f %.6f %.6f", classID, xCenter, yCenter, width, height)
		txtLines = append(txtLines, line)
	}

	txtContent := strings.Join(txtLines, "\n") + "\n"
	err := os.WriteFile(txtPath, []byte(txtContent), 0644)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal menyimpan file label: %v", err)})
		return
	}

	srcImgPath := filepath.Join("dataset", "raw", req.Filename)
	dstImgPath := filepath.Join("dataset", "images", "train", req.Filename)
	if _, err := os.Stat(srcImgPath); err == nil {
		err = moveFile(srcImgPath, dstImgPath)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal memindahkan file gambar: %v", err)})
			return
		}
	} else {
		// Jika file mentah tidak ada, pastikan file target sudah ada (mode edit)
		if _, err := os.Stat(dstImgPath); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "File gambar tidak ditemukan di antrean raw maupun training"})
			return
		}
	}

	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Berhasil menyimpan label"})
}

type TrainStartRequest struct {
	Epochs    int    `json:"epochs"`
	ModelType string `json:"model_type"`
	Device    string `json:"device"`
}

func handleAPITrainStart(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	trainingMu.Lock()
	if trainingActive {
		trainingMu.Unlock()
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Pelatihan AI sedang berjalan"})
		return
	}
	trainingMu.Unlock()

	var req TrainStartRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Invalid JSON"})
		return
	}

	if req.Epochs <= 0 {
		req.Epochs = 30
	}
	if req.ModelType != "yolo26" && req.ModelType != "yolov8" {
		req.ModelType = "yolo26"
	}
	if req.Device != "cpu" && req.Device != "gpu" {
		req.Device = "cpu"
	}

	// Verifikasi apakah dataset/data.yaml ada
	if _, err := os.Stat("dataset/data.yaml"); os.IsNotExist(err) {
		w.WriteHeader(http.StatusOK) // Kembalikan HTTP 200 dengan status error agar ditampilkan di browser
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "error",
			"message": "Gagal memulai pelatihan: Anda belum melabeli gambar apa pun! Silakan buka tab 'AI Labeling Center' dan beri label minimal 1 gambar terlebih dahulu.",
		})
		return
	}

	// Verifikasi apakah folder dataset/images/train memiliki gambar
	files, err := os.ReadDir("dataset/images/train")
	if err != nil || len(files) == 0 {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "error",
			"message": "Gagal memulai pelatihan: Folder latihan kosong! Harap masukkan foto di dataset/raw/ dan selesaikan pelabelan di tab 'AI Labeling Center'.",
		})
		return
	}

	os.Remove("snapshots/training.log")
	os.WriteFile("snapshots/training.log", []byte("Memulai inisialisasi pelatihan...\n"), 0644)

	baseModel := "yolo26n.pt"
	if req.ModelType == "yolov8" {
		baseModel = "yolov8n.pt"
	}
	
	pyDevice := "cpu"
	if req.Device == "gpu" {
		pyDevice = "0"
	}

	pyScript := fmt.Sprintf(`import sys
from ultralytics import YOLO

print("Memuat base model %s...")
model = YOLO("%s")

print("Mulai proses pelatihan...")
model.train(
    data="./dataset/data.yaml",
    epochs=%d,
    imgsz=640,
    device="%s",
    verbose=True
)
print("TRAINING_COMPLETED_SUCCESSFULLY")
`, baseModel, baseModel, req.Epochs, pyDevice)

	err = os.WriteFile("train_temp.py", []byte(pyScript), 0644)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Gagal membuat script pelatihan"})
		return
	}

	trainingMu.Lock()
	trainingActive = true
	trainingPercent = 0
	trainingMu.Unlock()

	go func() {
		defer func() {
			trainingMu.Lock()
			trainingActive = false
			trainingMu.Unlock()
			os.Remove("train_temp.py")
		}()

		cmd := exec.Command("py", "train_temp.py")
		
		trainingMu.Lock()
		trainingCmd = cmd
		trainingMu.Unlock()

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			appendTrainingLog("Gagal membuka stdout pipe proses: " + err.Error())
			return
		}
		cmd.Stderr = cmd.Stdout

		if err := cmd.Start(); err != nil {
			appendTrainingLog("Gagal memulai proses pelatihan: " + err.Error())
			return
		}

		appendTrainingLog("Proses Python berjalan (PID " + fmt.Sprintf("%d", cmd.Process.Pid) + ")...")

		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			appendTrainingLog(line)

			if strings.Contains(line, "/") {
				parts := strings.Fields(line)
				for _, part := range parts {
					if strings.Contains(part, "/") {
						subParts := strings.Split(part, "/")
						if len(subParts) == 2 {
							curEpoch, err1 := strconv.Atoi(subParts[0])
							totEpoch, err2 := strconv.Atoi(subParts[1])
							if err1 == nil && err2 == nil && totEpoch > 0 {
								percent := (curEpoch * 100) / totEpoch
								trainingMu.Lock()
								trainingPercent = percent
								trainingMu.Unlock()
							}
						}
					}
				}
			}
		}

		if err := cmd.Wait(); err != nil {
			appendTrainingLog("Pelatihan selesai dengan status error / dihentikan: " + err.Error())
			return
		}

		appendTrainingLog("Pelatihan Selesai 100%! Menyalin hasil model terbaik...")
		
		bestPtPath, errFind := findLatestBestPt()
		if errFind != nil {
			appendTrainingLog("Gagal menemukan file best.pt hasil latihan: " + errFind.Error())
			return
		}

		errCopy := moveFile(bestPtPath, "custom_model.pt")
		if errCopy != nil {
			appendTrainingLog("Gagal menyalin custom_model.pt: " + errCopy.Error())
			return
		}

		appendTrainingLog("Model kustom berhasil diaktifkan! Silakan restart layanan Python AI.")
		trainingMu.Lock()
		trainingPercent = 100
		trainingMu.Unlock()
	}()

	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Pelatihan dimulai"})
}

func handleAPITrainStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	trainingMu.Lock()
	active := trainingActive
	percent := trainingPercent
	trainingMu.Unlock()

	logBytes, err := os.ReadFile("snapshots/training.log")
	logText := ""
	if err == nil {
		logText = string(logBytes)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"is_training":      active,
		"progress_percent": percent,
		"logs":             logText,
	})
}

func handleAPITrainStop(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	trainingMu.Lock()
	defer trainingMu.Unlock()

	if !trainingActive || trainingCmd == nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Tidak ada pelatihan yang berjalan"})
		return
	}

	err := trainingCmd.Process.Kill()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Gagal menghentikan proses: " + err.Error()})
		return
	}

	trainingActive = false
	appendTrainingLog("\n=== PELATIHAN DIHENTIKAN OLEH PENGGUNA ===")
	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Pelatihan berhasil dihentikan"})
}

func handleAPILabelUpload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	// Batasi ukuran request ke 32MB untuk keamanan
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Gagal parsing form data: " + err.Error()})
		return
	}

	files := r.MultipartForm.File["files"]
	if len(files) == 0 {
		// Fallback ke single file key
		files = r.MultipartForm.File["file"]
	}

	if len(files) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Tidak ada file yang dikirim"})
		return
	}

	os.MkdirAll("dataset/raw", 0755)

	uploadedCount := 0
	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			continue
		}
		defer file.Close()

		// Buat nama file unik menggunakan timestamp agar tidak menimpa file lama
		ext := filepath.Ext(fileHeader.Filename)
		cleanExt := strings.ToLower(ext)
		if cleanExt != ".jpg" && cleanExt != ".jpeg" && cleanExt != ".png" {
			continue // Saring hanya gambar yang valid
		}

		uniqueName := fmt.Sprintf("raw_%s_%d%s", time.Now().Format("20060102_150405"), uploadedCount, cleanExt)
		dstPath := filepath.Join("dataset", "raw", uniqueName)

		dstFile, err := os.Create(dstPath)
		if err != nil {
			continue
		}
		defer dstFile.Close()

		_, err = io.Copy(dstFile, file)
		if err == nil {
			uploadedCount++
		}
		// Agar timestamp berbeda untuk iterasi berikutnya (milidetik)
		time.Sleep(10 * time.Millisecond)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "success",
		"message":  fmt.Sprintf("Berhasil mengunggah %d gambar", uploadedCount),
		"uploaded": uploadedCount,
	})
}

func handleAPILabelImagesLabeled(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	files, err := os.ReadDir("dataset/images/train")
	if err != nil {
		json.NewEncoder(w).Encode([]string{})
		return
	}

	var fileList []string
	for _, f := range files {
		if !f.IsDir() {
			ext := strings.ToLower(filepath.Ext(f.Name()))
			if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
				fileList = append(fileList, f.Name())
			}
		}
	}

	// Jika kosong, kembalikan array kosong bukan null
	if fileList == nil {
		fileList = []string{}
	}

	json.NewEncoder(w).Encode(fileList)
}

// GeminiAutoLabelResult merepresentasikan struktur data deteksi objek dari Gemini
type GeminiAutoLabelResult struct {
	Label      string    `json:"label"`
	Box        []float64 `json:"box"` // [x_min, y_min, x_max, y_max] normalisasi 0.0 - 1.0
	Confidence float64   `json:"confidence"`
}

// detectObjectsWithGemini memanggil API Gemini untuk mendeteksi objek 2D secara cerdas
func detectObjectsWithGemini(imgBytes []byte, apiKey string) ([]AIDetection, error) {
	// Encode gambar ke Base64
	base64Data := base64.StdEncoding.EncodeToString(imgBytes)

	prompt := `Identifikasi dan deteksi objek di dalam gambar ini. 
Deteksi secara akurat objek-objek berikut jika ada: 'manusia' (person), 'mobil' (car), 'motor' (motorcycle), 'api' (fire), 'asap' (smoke), 'pemadam api' (fire extinguisher), 'merokok' (smoking), 'tidur' (sleeping). 

Kembalikan hasil dalam format JSON terstruktur berbentuk array objek. Pastikan koordinat kotak pembatas (bounding box) berada dalam skala persentase normalisasi float dari 0.0 sampai 1.0 (di mana 0.0 adalah ujung kiri/atas, dan 1.0 adalah ujung kanan/bawah). 
Gunakan key koordinat "box" dengan format array: [x_min, y_min, x_max, y_max].
Tambahkan estimasi tingkat kepercayaan key "confidence" dari 0.0 sampai 1.0.
Jangan sertakan teks penjelasan lain atau wrapper markdown selain JSON yang valid.

Contoh format JSON yang diharapkan:
[
  {
    "label": "manusia",
    "box": [0.15, 0.20, 0.45, 0.85],
    "confidence": 0.95
  }
]`

	// Bentuk request payload
	reqPayload := GeminiRequest{
		Contents: []GeminiContent{
			{
				Parts: []GeminiPart{
					{
						Text: prompt,
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
		return nil, fmt.Errorf("failed to marshal Gemini request: %w", err)
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=%s", apiKey)

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-goog-api-key", apiKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call Gemini API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Gemini API error (%d): %s", resp.StatusCode, string(bodyBytes))
	}

	var geminiResp GeminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("empty response candidates")
	}

	rawText := geminiResp.Candidates[0].Content.Parts[0].Text
	cleanJSON := sanitizeJSON(rawText)

	var geminiDetections []GeminiAutoLabelResult
	if err := json.Unmarshal([]byte(cleanJSON), &geminiDetections); err != nil {
		return nil, fmt.Errorf("failed to parse detections JSON: %w", err)
	}

	// Dapatkan lebar dan tinggi asli gambar untuk mengonversikan koordinat persen ke koordinat piksel
	imgReader := bytes.NewReader(imgBytes)
	imgConfig, _, errConfig := image.DecodeConfig(imgReader)
	if errConfig != nil {
		return nil, fmt.Errorf("failed to read image dimensions: %w", errConfig)
	}
	width := float64(imgConfig.Width)
	height := float64(imgConfig.Height)

	var detections []AIDetection
	for _, gd := range geminiDetections {
		if len(gd.Box) < 4 {
			continue
		}
		
		// Konversikan koordinat normalisasi (0.0 - 1.0) menjadi koordinat absolut piksel gambar asli
		x1 := gd.Box[0] * width
		y1 := gd.Box[1] * height
		x2 := gd.Box[2] * width
		y2 := gd.Box[3] * height

		// Standardize label
		label := gd.Label
		if label == "manusia" {
			label = "person"
		} else if label == "api" {
			label = "fire"
		} else if label == "asap" {
			label = "smoke"
		}

		detections = append(detections, AIDetection{
			Class:      0,
			Label:      label,
			Confidence: gd.Confidence,
			Box:        []float64{x1, y1, x2, y2},
		})
	}

	return detections, nil
}

// handleAPILabelAutoDetect melakukan inferensi AI pada gambar dataset mentah/terlabeli untuk autolabeling
func handleAPILabelAutoDetect(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	filename := r.URL.Query().Get("filename")
	viewMode := r.URL.Query().Get("type") // "raw" atau "labeled"

	if filename == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Filename is required"})
		return
	}

	var imgPath string
	if viewMode == "labeled" {
		imgPath = filepath.Join("dataset", "images", "train", filename)
	} else {
		imgPath = filepath.Join("dataset", "raw", filename)
	}

	// Baca byte gambar
	imgBytes, err := os.ReadFile(imgPath)
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal membaca file gambar: %v", err)})
		return
	}

	// Cek provider VLM aktif
	configMu.RLock()
	vlmProvider := config.VLMProvider
	geminiKey := config.GeminiAPIKey
	openAIKey := config.OpenAIAPIKey
	configMu.RUnlock()

	var detections []AIDetection
	var errVLM error
	vlmUsed := false

	if vlmProvider == "openai" && openAIKey != "" {
		// Gunakan OpenAI Cloud VLM untuk auto-labeling cerdas
		detections, errVLM = detectObjectsWithOpenAI(imgBytes, openAIKey)
		if errVLM == nil {
			vlmUsed = true
		} else {
			log.Printf("Gagal auto-labeling via OpenAI, mencoba fallback ke model YOLO lokal: %v", errVLM)
		}
	} else if (vlmProvider == "gemini" || vlmProvider == "") && geminiKey != "" {
		// Gunakan Gemini Cloud VLM untuk auto-labeling cerdas
		detections, errVLM = detectObjectsWithGemini(imgBytes, geminiKey)
		if errVLM == nil {
			vlmUsed = true
		} else {
			log.Printf("Gagal auto-labeling via Gemini, mencoba fallback ke model YOLO lokal: %v", errVLM)
		}
	}

	if !vlmUsed {
		// Fallback ke deteksi lokal YOLOv8
		processorMu.Lock()
		if streamProcessor == nil || streamProcessor.aiClient == nil {
			processorMu.Unlock()
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Layanan AI Client lokal belum diinisialisasi"})
			return
		}
		aiClient := streamProcessor.aiClient
		processorMu.Unlock()

		var errLocal error
		detections, errLocal = aiClient.Detect(imgBytes)
		if errLocal != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal mendeteksi via AI lokal: %v", errLocal)})
			return
		}
	}

	// Kembalikan daftar deteksi
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "success",
		"detections": detections,
	})
}

// handleAPITestOpenAI memvalidasi API Key OpenAI ke server OpenAI secara langsung dengan query ringan
func handleAPITestOpenAI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		http.Error(w, `{"status": "error", "message": "Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var reqBody struct {
		OpenAIAPIKey string `json:"openai_api_key"`
	}
	err := json.NewDecoder(r.Body).Decode(&reqBody)
	if err != nil || reqBody.OpenAIAPIKey == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "API Key tidak boleh kosong"})
		return
	}

	// Payload request ringan hanya kirim teks pendek untuk verifikasi validitas key
	testPayloadClean := map[string]interface{}{
		"model": "gpt-4o-mini",
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": "Hello! Reply with YES if this API key works.",
			},
		},
		"max_tokens": 5,
	}

	jsonBytes, err := json.Marshal(testPayloadClean)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal menyusun payload: %v", err)})
		return
	}

	url := "https://api.openai.com/v1/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal inisialisasi request HTTP: %v", err)})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", reqBody.OpenAIAPIKey))

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": fmt.Sprintf("Gagal terhubung ke OpenAI API (%v)", err)})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "error",
			"message": fmt.Sprintf("Kunci API tidak valid (Status HTTP %d)", resp.StatusCode),
		})
		return
	}

	// Jika status 200 OK, berarti kunci API valid!
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Koneksi sukses! API Key Valid.",
	})
}

// detectObjectsWithOpenAI memanggil API OpenAI (gpt-4o-mini) untuk mendeteksi objek 2D secara cerdas
func detectObjectsWithOpenAI(imgBytes []byte, apiKey string) ([]AIDetection, error) {
	// Encode gambar ke Base64
	base64Data := base64.StdEncoding.EncodeToString(imgBytes)

	prompt := `Identifikasi dan deteksi objek di dalam gambar ini. 
Deteksi secara akurat objek-objek berikut jika ada: 'manusia' (person), 'mobil' (car), 'motor' (motorcycle), 'api' (fire), 'asap' (smoke), 'pemadam api' (fire extinguisher), 'merokok' (smoking), 'tidur' (sleeping). 

Kembalikan hasil dalam format JSON terstruktur berbentuk array objek. Pastikan koordinat kotak pembatas (bounding box) berada dalam skala persentase normalisasi float dari 0.0 sampai 1.0 (di mana 0.0 adalah ujung kiri/atas, dan 1.0 adalah ujung kanan/bawah). 
Gunakan key koordinat "box" dengan format array: [x_min, y_min, x_max, y_max].
Tambahkan estimasi tingkat kepercayaan key "confidence" dari 0.0 sampai 1.0.
Jangan sertakan teks penjelasan lain atau wrapper markdown selain JSON yang valid.

Contoh format JSON yang diharapkan:
[
  {
    "label": "manusia",
    "box": [0.15, 0.20, 0.45, 0.85],
    "confidence": 0.95
  }
]`

	// Payload request
	reqPayload := OpenAIRequest{
		Model: "gpt-4o-mini",
		Messages: []OpenAIMessage{
			{
				Role: "user",
				Content: []OpenAIContent{
					{
						Type: "text",
						Text: prompt,
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

	jsonBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal OpenAI request: %w", err)
	}

	url := "https://api.openai.com/v1/chat/completions"

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call OpenAI API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error (%d): %s", resp.StatusCode, string(bodyBytes))
	}

	var openAIResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(openAIResp.Choices) == 0 {
		return nil, fmt.Errorf("empty response choices")
	}

	rawText := openAIResp.Choices[0].Message.Content
	cleanJSON := sanitizeJSON(rawText)

	var geminiDetections []GeminiAutoLabelResult
	if err := json.Unmarshal([]byte(cleanJSON), &geminiDetections); err != nil {
		return nil, fmt.Errorf("failed to parse detections JSON: %w", err)
	}

	// Dapatkan dimensi asli gambar
	imgReader := bytes.NewReader(imgBytes)
	imgConfig, _, errConfig := image.DecodeConfig(imgReader)
	if errConfig != nil {
		return nil, fmt.Errorf("failed to read image dimensions: %w", errConfig)
	}
	width := float64(imgConfig.Width)
	height := float64(imgConfig.Height)

	var detections []AIDetection
	for _, gd := range geminiDetections {
		if len(gd.Box) < 4 {
			continue
		}
		
		x1 := gd.Box[0] * width
		y1 := gd.Box[1] * height
		x2 := gd.Box[2] * width
		y2 := gd.Box[3] * height

		label := gd.Label
		if label == "manusia" {
			label = "person"
		} else if label == "api" {
			label = "fire"
		} else if label == "asap" {
			label = "smoke"
		}

		detections = append(detections, AIDetection{
			Class:      0,
			Label:      label,
			Confidence: gd.Confidence,
			Box:        []float64{x1, y1, x2, y2},
		})
	}

	return detections, nil
}

// handleAPIVLMTrigger memicu pemanggilan VLM secara manual dari halaman web dashboard
func handleAPIVLMTrigger(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		http.Error(w, `{"status": "error", "message": "Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	processorMu.Lock()
	if streamProcessor == nil {
		processorMu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Streaming processor belum siap"})
		return
	}
	sp := streamProcessor
	processorMu.Unlock()

	// Ambil frame JPEG terkini
	sp.currentFrameMu.RLock()
	rawJPEG := make([]byte, len(sp.currentFrame))
	copy(rawJPEG, sp.currentFrame)
	sp.currentFrameMu.RUnlock()

	if len(rawJPEG) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "Frame kamera belum tersedia"})
		return
	}

	// Pemicu pemanggilan API kognitif secara manual
	sp.geminiBusyMu.Lock()
	gBusy := sp.geminiBusy
	sp.geminiBusyMu.Unlock()

	if gBusy {
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": "AI sedang sibuk memproses analisis"})
		return
	}

	// Panggil VLM secara asinkron
	sp.geminiLastCheck = time.Now()
	go sp.callVLMAPI(rawJPEG)

	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Kueri VLM berhasil dipicu secara manual"})
}
