package main

import (
	"database/sql"
	"fmt"
	"time"

	_ "modernc.org/sqlite"
)

// Detection merepresentasikan baris rekaman hasil deteksi di database
type Detection struct {
	ID           int       `json:"id"`
	Timestamp    time.Time `json:"timestamp"`
	Label        string    `json:"label"`
	Confidence   float64   `json:"confidence"`
	SnapshotPath string    `json:"snapshot_path"`
}

// Stats merepresentasikan statistik aktivitas deteksi cctv
type Stats struct {
	TotalDetections int `json:"total_detections"`
	DetectionsToday int `json:"detections_today"`
}

// DBManager mengelola koneksi database SQLite
type DBManager struct {
	db *sql.DB
}

// NewDBManager menginisialisasi database SQLite dan membuat tabel jika belum ada
func NewDBManager(dbPath string) (*DBManager, error) {
	// Membuka database dengan driver "sqlite" dari modernc.org/sqlite
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Buat tabel jika belum ada
	schema := `
	CREATE TABLE IF NOT EXISTS detections (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
		label TEXT NOT NULL,
		confidence REAL NOT NULL,
		snapshot_path TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
	`
	_, err = db.Exec(schema)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create schema: %w", err)
	}

	return &DBManager{db: db}, nil
}

// Close menutup koneksi database
func (m *DBManager) Close() error {
	if m.db != nil {
		return m.db.Close()
	}
	return nil
}

// LogDetection mencatat aktivitas deteksi objek baru ke database
func (m *DBManager) LogDetection(label string, confidence float64, snapshotPath string) (int64, error) {
	query := `INSERT INTO detections (timestamp, label, confidence, snapshot_path) VALUES (?, ?, ?, ?)`

	// Gunakan waktu lokal Indonesia (WIB/WITA/WIT sesuai server)
	now := time.Now()

	res, err := m.db.Exec(query, now, label, confidence, snapshotPath)
	if err != nil {
		return 0, fmt.Errorf("failed to log detection: %w", err)
	}

	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	return id, nil
}

// GetRecentDetections mengambil riwayat deteksi terbaru berdasarkan limit
func (m *DBManager) GetRecentDetections(limit int) ([]Detection, error) {
	query := `SELECT id, timestamp, label, confidence, snapshot_path FROM detections ORDER BY timestamp DESC LIMIT ?`
	rows, err := m.db.Query(query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent detections: %w", err)
	}
	defer rows.Close()

	var detections []Detection
	for rows.Next() {
		var d Detection
		var tsStr string
		err := rows.Scan(&d.ID, &tsStr, &d.Label, &d.Confidence, &d.SnapshotPath)
		if err != nil {
			return nil, err
		}

		// Konversi string timestamp SQLite ke time.Time
		t, err := time.Parse("2006-01-02 15:04:05.999999999-07:00", tsStr)
		if err != nil {
			// Coba parse dengan format UTC standar jika format lokal berbeda
			t, err = time.Parse(time.RFC3339, tsStr)
			if err != nil {
				// Fallback jika format tidak standar
				t = time.Now()
			}
		}
		d.Timestamp = t
		detections = append(detections, d)
	}

	// Check for any errors that occurred during iteration
	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error during rows iteration: %w", err)
	}

	return detections, nil
}

// GetStats mengambil rangkuman statistik untuk ditampilkan di dashboard
func (m *DBManager) GetStats() (Stats, error) {
	var stats Stats

	// Hitung total deteksi
	err := m.db.QueryRow(`SELECT COUNT(*) FROM detections`).Scan(&stats.TotalDetections)
	if err != nil {
		return stats, fmt.Errorf("failed to count total detections: %w", err)
	}

	// Hitung deteksi hari ini
	todayStr := time.Now().Format("2006-01-02")
	err = m.db.QueryRow(
		`SELECT COUNT(*) FROM detections WHERE date(timestamp) = date(?)`,
		todayStr,
	).Scan(&stats.DetectionsToday)
	if err != nil {
		return stats, fmt.Errorf("failed to count detections today: %w", err)
	}

	return stats, nil
}
