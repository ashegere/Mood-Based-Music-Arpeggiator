import React, { useState, useEffect } from 'react';
import { Music, LogOut, Sparkles, Play, Download, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

const Home = () => {
  const navigate = useNavigate();
  const [tempo, setTempo] = useState(120);
  const [selectedKey, setSelectedKey] = useState('D');
  const [scale, setScale] = useState('Major');
  const [noteCount, setNoteCount] = useState(12);
  const [mood, setMood] = useState('');
  const [selectedInstrument, setSelectedInstrument] = useState('Piano');
  const [hasGenerated, setHasGenerated] = useState(false);

  // Mock data for visualization
  const [patternData, setPatternData] = useState({
    notes: ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'F5'],
    heights: [65, 72, 75, 78, 82, 85, 88, 92, 95, 98, 100, 95],
    noteRange: 'C4 - B5',
    avgVelocity: 91,
    patternLength: 12
  });

  const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const scales = ['Major', 'Minor', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian', 'Locrian'];
  const instruments = [
    { name: 'Synth Pluck', icon: '🎵' },
    { name: 'Piano', icon: '🎹' },
    { name: 'Guitar', icon: '🎸' },
    { name: 'Harp', icon: '🎼' }
  ];

  const handleGenerate = () => {
    setHasGenerated(true);
    // TODO: Call API to generate arpeggio
  };

  useEffect(() => {
    // Redirect to landing page if user is not logged in
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/');
    }
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/');
  };

  return (
    <div className="home-page">
      <header className="home-header">
        <div className="home-logo">
          <Music size={24} strokeWidth={3} />
          <span>arpeggiator.ai</span>
        </div>
        <div className="home-nav">
          <button className="nav-btn">
            <Sparkles size={18} />
            Build
          </button>
          <button className="nav-btn" onClick={handleLogout}>
            <LogOut size={18} />
            Log out
          </button>
        </div>
      </header>

      <main className="home-content">
        {/* Left Panel - Build Your Arpeggio */}
        <div className="panel build-panel">
          <div className="panel-header">
            <span className="panel-icon">✨</span>
            <h2>Build Your Arpeggio</h2>
          </div>

          <div className="control-group">
            <label>Tempo (BPM)</label>
            <div className="slider-container">
              <div className="slider-track">
                <div className="slider-fill" style={{ width: `${((tempo - 60) / (200 - 60)) * 100}%` }} />
                <input
                  type="range"
                  min="60"
                  max="200"
                  value={tempo}
                  onChange={(e) => setTempo(e.target.value)}
                  className="slider"
                />
              </div>
              <span className="slider-value">{tempo}</span>
            </div>
          </div>

          <div className="control-group">
            <label>Key</label>
            <div className="key-grid">
              {keys.map((key) => (
                <button
                  key={key}
                  className={`key-btn ${selectedKey === key ? 'selected' : ''}`}
                  onClick={() => setSelectedKey(key)}
                >
                  {key}
                </button>
              ))}
            </div>
          </div>

          <div className="control-group">
            <label>Scale</label>
            <div className="select-wrapper">
              <select
                value={scale}
                onChange={(e) => setScale(e.target.value)}
                className="scale-select"
              >
                {scales.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <ChevronDown className="select-icon" size={20} />
            </div>
          </div>

          <div className="control-group">
            <label>Number of Notes in Pattern</label>
            <div className="slider-container">
              <div className="slider-track">
                <div className="slider-fill" style={{ width: `${((noteCount - 4) / (32 - 4)) * 100}%` }} />
                <input
                  type="range"
                  min="4"
                  max="32"
                  value={noteCount}
                  onChange={(e) => setNoteCount(e.target.value)}
                  className="slider"
                />
              </div>
              <span className="slider-value">{noteCount}</span>
            </div>
          </div>

          <div className="control-group">
            <label>Mood-Based Keyword/Phrase</label>
            <textarea
              value={mood}
              onChange={(e) => setMood(e.target.value)}
              placeholder="e.g., happy and energetic, calm and peaceful, dark and mysterious..."
              className="mood-textarea"
              rows={4}
            />
          </div>

          <button className="generate-btn" onClick={handleGenerate}>
            <Sparkles size={20} />
            Generate Arpeggio
          </button>
        </div>

        {/* Right Panel - Export and Preview */}
        <div className="panel preview-panel">
          <div className="panel-header">
            <span className="panel-icon">▶</span>
            <h2>Export and Preview</h2>
          </div>

          <div className="visualization-section">
            <label>Pattern Sequence Visualization</label>

            {hasGenerated ? (
              <>
                <div className="bar-chart">
                  {patternData.heights.map((height, index) => (
                    <div key={index} className="bar-container">
                      <div
                        className="bar"
                        style={{ height: `${height}%` }}
                      />
                      <span className="bar-label">{patternData.notes[index]}</span>
                    </div>
                  ))}
                </div>

                <div className="waveform">
                  <svg width="100%" height="100%" viewBox="0 0 400 60" preserveAspectRatio="none">
                    <polyline
                      points="0,30 40,25 80,35 120,20 160,40 200,15 240,38 280,22 320,33 360,28 400,30"
                      fill="none"
                      strokeWidth="2"
                    />
                  </svg>
                </div>

                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Pattern Length</div>
                    <div className="stat-value primary">{patternData.patternLength} notes</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Note Range</div>
                    <div className="stat-value">{patternData.noteRange}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Avg Velocity</div>
                    <div className="stat-value">{patternData.avgVelocity}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Tempo</div>
                    <div className="stat-value">{tempo} BPM</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">☀</div>
                <p>Generate a pattern to see the preview</p>
              </div>
            )}
          </div>

          <div className="control-group">
            <label>Instrument Mode</label>
            <div className="instrument-grid">
              {instruments.map((instrument) => (
                <button
                  key={instrument.name}
                  className={`instrument-btn ${selectedInstrument === instrument.name ? 'selected' : ''}`}
                  onClick={() => setSelectedInstrument(instrument.name)}
                >
                  <span className="instrument-icon">{instrument.icon}</span>
                  {instrument.name}
                </button>
              ))}
            </div>
          </div>

          <div className="action-buttons">
            <button className="preview-btn">
              <Play size={18} />
              Preview
            </button>
            <button className="export-btn">
              <Download size={18} />
              Export as MP4
              <ChevronDown size={16} />
            </button>
          </div>
        </div>
      </main>

      <footer className="home-footer">
        <div className="footer-text">© 2025 Arpeggiator.ai - Powered by AI</div>
        <div className="footer-links">
          <button className="footer-btn">☕ Buy me a coffee</button>
          <button className="footer-btn">in</button>
          <button className="footer-btn">🎵</button>
          <button className="footer-btn">✉</button>
        </div>
      </footer>
    </div>
  );
};

export default Home;
