package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"
)

// AIDetection merepresentasikan data deteksi objek dari model YOLO
type AIDetection struct {
	Class      int       `json:"class"`
	Label      string    `json:"label"`
	Confidence float64   `json:"confidence"`
	Box        []float64 `json:"box"` // [x1, y1, x2, y2]
}

// AIResponse merepresentasikan payload respons dari API Python
type AIResponse struct {
	Detections []AIDetection `json:"detections"`
}

// AIClient mengelola komunikasi HTTP dengan server inferensi YOLOv8
type AIClient struct {
	Endpoint   string
	httpClient *http.Client
}

// NewAIClient membuat instansi baru AIClient
func NewAIClient(endpoint string) *AIClient {
	return &AIClient{
		Endpoint: endpoint,
		httpClient: &http.Client{
			Timeout: 3 * time.Second, // Timeout cepat agar streaming tidak lambat jika model offline
		},
	}
}

// IsOnline mengecek apakah layanan Python YOLOv8 sedang aktif
func (c *AIClient) IsOnline() bool {
	resp, err := c.httpClient.Get(c.Endpoint + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Detect mengirim frame JPEG ke layanan YOLOv8 dan mengembalikan daftar objek yang terdeteksi
func (c *AIClient) Detect(jpegData []byte) ([]AIDetection, error) {
	// Membuat request multipart form-data
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	
	// Tambahkan field file dengan byte JPEG
	part, err := writer.CreateFormFile("file", "frame.jpg")
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	
	_, err = part.Write(jpegData)
	if err != nil {
		return nil, fmt.Errorf("failed to write frame to buffer: %w", err)
	}
	
	err = writer.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Buat request POST
	req, err := http.NewRequest("POST", c.Endpoint+"/detect", body)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Jalankan request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to communicate with AI server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("AI server returned error (%d): %s", resp.StatusCode, string(respBody))
	}

	// Baca dan decode respons JSON
	var aiResp AIResponse
	err = json.NewDecoder(resp.Body).Decode(&aiResp)
	if err != nil {
		return nil, fmt.Errorf("failed to parse AI response: %w", err)
	}

	return aiResp.Detections, nil
}
