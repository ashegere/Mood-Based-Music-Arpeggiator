import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Music, Zap, Users, Sparkles, ChevronRight, Play, Settings, Download, Hash } from 'lucide-react';

export default function ArpeggiatorLanding() {
  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedData, setGeneratedData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Redirect to home if user is already logged in
    const token = localStorage.getItem('token');
    if (token) {
      navigate('/home');
      return;
    }
    setIsVisible(true);
  }, [navigate]);

  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const [activeNotes, setActiveNotes] = useState([0, 4, 7]); // C major chord
  const [patternLength, setPatternLength] = useState('8');
  const [mood, setMood] = useState('energetic');
  const [instructions, setInstructions] = useState('');

  
  return (
    <div className="landing-page">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Exo+2:wght@700;800;900&display=swap');

        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          overflow-x: hidden;
        }

        .landing-page {
          font-family: 'Inter', sans-serif;
          color: #ffffff;
          background: #0e1225;
          min-height: 100vh;
          position: relative;
          overflow-x: hidden;
        }

        .landing-page::before {
          content: '';
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background:
            radial-gradient(ellipse 60% 50% at 10% 20%, rgba(138, 43, 226, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 50% 60% at 90% 80%, rgba(255, 107, 0, 0.08) 0%, transparent 60%);
          pointer-events: none;
          z-index: 0;
        }

        .container {
          max-width: 1400px;
          margin: 0 auto;
          padding: 0 40px;
          position: relative;
          z-index: 1;
        }

        /* Header */
        .header {
          padding: 30px 0;
          opacity: 0;
          animation: fadeInDown 0.8s ease forwards;
        }

        .logo {
          display: flex;
          align-items: center;
          gap: 10px;
          font-family: 'Exo 2', sans-serif;
          font-size: 20px;
          font-weight: 800;
          letter-spacing: -0.5px;
        }

        .logo-icon {
          width: 32px;
          height: 32px;
          background: linear-gradient(135deg, #ff6b00 0%, #ff8c3a 100%);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .nav {
          display: flex;
          gap: 50px;
          align-items: center;
        }

        .nav-link {
          color: rgba(255, 255, 255, 0.8);
          text-decoration: none;
          font-size: 15px;
          font-weight: 500;
          transition: all 0.3s ease;
          position: relative;
        }

        .nav-link:hover {
          color: #ff6b00;
        }

        .nav-link::after {
          content: '';
          position: absolute;
          bottom: -5px;
          left: 0;
          width: 0;
          height: 2px;
          background: #ff6b00;
          transition: width 0.3s ease;
        }

        .nav-link:hover::after {
          width: 100%;
        }

        /* Hero Section */
        .hero {
          padding: 80px 0 120px;
          text-align: center;
          opacity: 0;
          animation: fadeInUp 1s ease 0.2s forwards;
        }

        .hero-title {
          font-family: 'Exo 2', sans-serif;
          font-size: clamp(48px, 8vw, 88px);
          font-weight: 900;
          line-height: 1.1;
          margin-bottom: 40px;
          letter-spacing: -2px;
        }

        .hero-title .energetic {
          color: #ff6b00;
          display: block;
          text-shadow: 0 0 40px rgba(255, 107, 0, 0.5);
        }

        /* Arpeggiator Interface */
        .arpeggiator-demo {
          max-width: 900px;
          margin: 60px auto;
          background: rgba(12, 15, 30, 0.75);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border-radius: 20px;
          padding: 40px;
          border: 1px solid rgba(255, 255, 255, 0.07);
          box-shadow:
            0 0 0 1px rgba(255, 107, 0, 0.04),
            0 24px 64px rgba(0, 0, 0, 0.6);
          opacity: 0;
          animation: fadeInScale 1s ease 0.4s forwards;
          position: relative;
          overflow: hidden;
        }

        .arpeggiator-demo::before {
          content: '';
          position: absolute;
          top: 0;
          left: 15%;
          right: 15%;
          height: 1px;
          background: linear-gradient(90deg,
            transparent,
            rgba(255, 107, 0, 0.7),
            rgba(138, 43, 226, 0.5),
            transparent);
        }

        .note-grid {
          display: grid;
          grid-template-columns: repeat(12, 1fr);
          gap: 8px;
          margin-bottom: 30px;
        }

        .note-button {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px 8px;
          color: rgba(255, 255, 255, 0.6);
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          font-family: 'Inter', sans-serif;
        }

        .note-button:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 107, 0, 0.3);
        }

        .note-button.active {
          background: linear-gradient(135deg, #ff6b00 0%, #ff8c3a 100%);
          border-color: #ff6b00;
          color: #ffffff;
          box-shadow: 0 4px 20px rgba(255, 107, 0, 0.4);
        }

        .control-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-bottom: 30px;
        }

        .input-group {
          position: relative;
        }

        .input-label {
          display: block;
          font-size: 12px;
          color: rgba(255, 255, 255, 0.5);
          margin-bottom: 8px;
          font-weight: 500;
        }

        .input-field {
          width: 100%;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 12px 16px;
          color: #ffffff;
          font-size: 14px;
          font-family: 'Inter', sans-serif;
          transition: all 0.3s ease;
        }

        .input-field:focus {
          outline: none;
          border-color: #ff6b00;
          box-shadow: 0 0 0 3px rgba(255, 107, 0, 0.1);
        }

        .textarea-field {
          min-height: 100px;
          resize: vertical;
        }

        .generate-btn {
          width: 100%;
          background: linear-gradient(135deg, #ff6b00 0%, #ff8c3a 100%);
          border: none;
          border-radius: 12px;
          padding: 18px 32px;
          color: #ffffff;
          font-size: 16px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          font-family: 'Inter', sans-serif;
          box-shadow: 0 8px 30px rgba(255, 107, 0, 0.3);
        }

        .generate-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 40px rgba(255, 107, 0, 0.5);
        }

        .generate-btn:active {
          transform: translateY(0);
        }

        .visualization-area {
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 16px;
          padding: 60px;
          text-align: center;
          color: rgba(255, 255, 255, 0.3);
          font-size: 14px;
          margin: 30px 0;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .action-buttons {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
        }

        .action-btn {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 10px;
          padding: 14px 20px;
          color: #ffffff;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          font-family: 'Inter', sans-serif;
        }

        .action-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 107, 0, 0.3);
        }

        .cta-button {
          background: linear-gradient(135deg, #ff6b00, #ff8c3a);
          border: none;
          border-radius: 12px;
          padding: 16px 40px;
          color: #ffffff;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          margin-top: 40px;
          display: inline-flex;
          align-items: center;
          gap: 10px;
          font-family: 'Inter', sans-serif;
          box-shadow: 0 8px 30px rgba(255, 107, 0, 0.28);
          text-decoration: none;
        }

        .cta-button:hover {
          opacity: 0.92;
          transform: translateY(-2px);
          box-shadow: 0 12px 40px rgba(255, 107, 0, 0.4);
        }

        /* Features Section */
        .features-section {
          padding: 100px 0;
          opacity: 0;
          animation: fadeInUp 1s ease 0.6s forwards;
        }

        .section-title {
          font-family: 'Exo 2', sans-serif;
          font-size: clamp(36px, 5vw, 52px);
          font-weight: 900;
          text-align: center;
          margin-bottom: 80px;
          letter-spacing: -1px;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 24px;
          margin-bottom: 80px;
        }

        .feature-card {
          background: rgba(12, 15, 30, 0.75);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 20px;
          padding: 40px;
          transition: all 0.4s ease;
          position: relative;
          overflow: hidden;
        }

        .feature-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 15%;
          right: 15%;
          height: 1px;
          background: linear-gradient(90deg,
            transparent,
            rgba(255, 107, 0, 0.7),
            rgba(138, 43, 226, 0.5),
            transparent);
        }

        .feature-card:hover {
          transform: translateY(-8px);
          border-color: rgba(255, 107, 0, 0.25);
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }

        .feature-card.large {
          grid-column: span 1;
          min-height: 320px;
        }

        .feature-title {
          font-family: 'Exo 2', sans-serif;
          font-size: 24px;
          font-weight: 800;
          margin-bottom: 12px;
        }

        .feature-description {
          color: rgba(255, 255, 255, 0.7);
          font-size: 15px;
          line-height: 1.6;
          margin-bottom: 24px;
        }

        .feature-image {
          width: 100%;
          height: 180px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 12px;
          margin-top: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: rgba(255, 255, 255, 0.3);
          font-size: 14px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .feature-badge {
          background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          padding: 6px 16px;
          border-radius: 20px;
          font-size: 13px;
          font-weight: 600;
          display: inline-block;
          margin-bottom: 20px;
        }

        .feature-icons {
          display: flex;
          gap: 20px;
          margin-top: 30px;
        }

        .icon-box {
          width: 80px;
          height: 80px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 32px;
          font-weight: 900;
          font-family: 'Exo 2', sans-serif;
        }

        /* Experience Section */
        .experience-section {
          padding: 100px 0;
          opacity: 0;
          animation: fadeInUp 1s ease 0.8s forwards;
        }

        .experience-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 60px;
          align-items: center;
        }

        .experience-content h2 {
          font-family: 'Exo 2', sans-serif;
          font-size: clamp(36px, 5vw, 48px);
          font-weight: 900;
          margin-bottom: 24px;
          letter-spacing: -1px;
          line-height: 1.2;
        }

        .experience-content p {
          color: rgba(255, 255, 255, 0.7);
          font-size: 16px;
          line-height: 1.7;
        }

        .experience-visual {
          background: rgba(12, 15, 30, 0.75);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 20px;
          padding: 60px;
          min-height: 400px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 20px;
          position: relative;
          overflow: hidden;
        }

        .experience-visual::before {
          content: '';
          position: absolute;
          top: 0;
          left: 15%;
          right: 15%;
          height: 1px;
          background: linear-gradient(90deg,
            transparent,
            rgba(255, 107, 0, 0.7),
            rgba(138, 43, 226, 0.5),
            transparent);
        }

        .analysis-card {
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 20px 24px;
          display: flex;
          align-items: center;
          gap: 16px;
          transition: all 0.3s ease;
        }

        .analysis-card:hover {
          background: rgba(0, 0, 0, 0.6);
          border-color: #ff6b00;
        }

        .analysis-icon {
          width: 48px;
          height: 48px;
          background: linear-gradient(135deg, #ff6b00 0%, #ff8c3a 100%);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .analysis-text {
          font-weight: 600;
          font-size: 15px;
        }

        /* Founder Section */
        .founder-section {
          padding: 100px 0;
          opacity: 0;
          animation: fadeInUp 1s ease 1s forwards;
        }

        .founder-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 80px;
          align-items: center;
        }

        .founder-card {
          background: rgba(12, 15, 30, 0.75);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 20px;
          padding: 40px;
          display: flex;
          gap: 24px;
          align-items: center;
          position: relative;
          overflow: hidden;
        }

        .founder-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 15%;
          right: 15%;
          height: 1px;
          background: linear-gradient(90deg,
            transparent,
            rgba(255, 107, 0, 0.7),
            rgba(138, 43, 226, 0.5),
            transparent);
        }

        .founder-image {
          width: 100px;
          height: 100px;
          background: linear-gradient(135deg, #ff6b00 0%, #ff8c3a 100%);
          border-radius: 50%;
          flex-shrink: 0;
        }

        .founder-info h3 {
          font-family: 'Exo 2', sans-serif;
          font-size: 24px;
          font-weight: 800;
          margin-bottom: 4px;
        }

        .founder-info .role {
          color: rgba(255, 255, 255, 0.6);
          font-size: 14px;
          margin-bottom: 12px;
        }

        .founder-info .badge {
          background: rgba(255, 107, 0, 0.2);
          color: #ff6b00;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
          display: inline-block;
        }

        .founder-content h2 {
          font-family: 'Exo 2', sans-serif;
          font-size: clamp(36px, 5vw, 48px);
          font-weight: 900;
          margin-bottom: 24px;
          letter-spacing: -1px;
          line-height: 1.2;
        }

        .founder-content p {
          color: rgba(255, 255, 255, 0.7);
          font-size: 16px;
          line-height: 1.7;
        }

        /* Trusted Section */
        .trusted-section {
          padding: 80px 0;
          text-align: center;
          opacity: 0;
          animation: fadeInUp 1s ease 1.2s forwards;
        }

        .trusted-title {
          font-family: 'Exo 2', sans-serif;
          font-size: 28px;
          font-weight: 800;
          margin-bottom: 50px;
        }

        .trusted-logos {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 60px;
          flex-wrap: wrap;
        }

        .logo-item {
          opacity: 0.5;
          transition: opacity 0.3s ease;
          font-size: 14px;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.6);
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .logo-item:hover {
          opacity: 1;
        }

        /* Testimonial */
        .testimonial-section {
          padding: 100px 0;
          opacity: 0;
          animation: fadeInUp 1s ease 1.4s forwards;
        }

        .testimonial {
          max-width: 1000px;
          margin: 0 auto;
          background: rgba(12, 15, 30, 0.75);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 20px;
          padding: 60px;
          text-align: center;
          position: relative;
          overflow: hidden;
        }

        .testimonial::before {
          content: '';
          position: absolute;
          top: 0;
          left: 15%;
          right: 15%;
          height: 1px;
          background: linear-gradient(90deg,
            transparent,
            rgba(255, 107, 0, 0.7),
            rgba(138, 43, 226, 0.5),
            transparent);
        }

        .testimonial-text {
          font-family: 'Exo 2', sans-serif;
          font-size: clamp(24px, 3vw, 36px);
          font-weight: 700;
          line-height: 1.5;
          margin-bottom: 40px;
          color: rgba(255, 255, 255, 0.95);
        }

        .testimonial-author {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
        }

        .author-image {
          width: 56px;
          height: 56px;
          background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          border-radius: 50%;
        }

        .author-info {
          text-align: left;
        }

        .author-name {
          font-weight: 700;
          font-size: 16px;
          margin-bottom: 2px;
        }

        .author-title {
          color: rgba(255, 255, 255, 0.5);
          font-size: 14px;
        }

        /* Footer */
        .footer {
          padding: 40px 0;
          text-align: center;
          border-top: 1px solid rgba(255, 255, 255, 0.05);
          opacity: 0;
          animation: fadeInUp 1s ease 1.6s forwards;
        }

        .footer-content {
          display: flex;
          justify-content: space-between;
          align-items: center;
          color: rgba(255, 255, 255, 0.4);
          font-size: 14px;
        }

        .footer-links {
          display: flex;
          gap: 30px;
        }

        .footer-link {
          color: rgba(255, 255, 255, 0.4);
          text-decoration: none;
          transition: color 0.3s ease;
        }

        .footer-link:hover {
          color: #ff6b00;
        }

        /* Animations */
        @keyframes fadeInDown {
          from {
            opacity: 0;
            transform: translateY(-30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInScale {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        /* Responsive */
        @media (max-width: 1024px) {
          .features-grid {
            grid-template-columns: 1fr;
          }

          .experience-grid,
          .founder-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 768px) {
          .container {
            padding: 0 20px;
          }

          .nav {
            display: none;
          }

          .note-grid {
            grid-template-columns: repeat(6, 1fr);
          }

          .control-row {
            grid-template-columns: 1fr;
          }

          .action-buttons {
            grid-template-columns: 1fr;
          }

          .arpeggiator-demo {
            padding: 24px;
          }

          .testimonial {
            padding: 40px 24px;
          }

          .footer-content {
            flex-direction: column;
            gap: 20px;
          }
        }
      `}</style>

      {/* Header */}
      <header className="header">
        <div className="container">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="logo">
              <div className="logo-icon">
                <Music size={18} strokeWidth={3} />
              </div>
              arpeggiator.ai
            </div>
            <nav className="nav">
              <Link to="/login" className="nav-link">Log in</Link>
            </nav>
          </div>
        </div>
      </header>

      <div className="container">
        {/* Hero Section */}
        <section className="hero">
          <h1 className="hero-title">
            Make your music
            <span className="energetic">energetic.</span>
          </h1>
          <Link to="/login" className="cta-button">
            Start Building
            <ChevronRight size={20} />
          </Link>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <h2 className="section-title">Why Choose ArpeggiateAI</h2>
          
          <div className="features-grid">
            <div className="feature-card large">
              <h3 className="feature-title">Emotion-driven Patterns</h3>
              <p className="feature-description">
                Generate evolving patterns based on emotional input.
              </p>
              <div className="feature-image">
                <div style={{ textAlign: 'center' }}>
                  <Sparkles size={48} style={{ color: '#ff6b00', marginBottom: '12px' }} />
                  <div style={{ fontSize: '16px', fontWeight: 600, color: 'rgba(255,255,255,0.7)' }}>
                    Emotion Analysis
                  </div>
                </div>
              </div>
            </div>

            <div className="feature-card large">
              <h3 className="feature-title">Overcome "Producer's Block"</h3>
              <p className="feature-description">
                Generate a suggestive and totally original compositions to continue working on.
              </p>
              <div className="feature-image" style={{ 
                background: 'linear-gradient(135deg, rgba(255,107,0,0.1) 0%, rgba(99,102,241,0.1) 100%)',
              }}>
                <Music size={48} style={{ color: '#ff6b00' }} />
              </div>
            </div>

            <div className="feature-card">
              <h3 className="feature-title">User-Friendly Interface</h3>
              <p className="feature-description">
                Easy-to-use interface for seamless music creation.
              </p>
              <div className="feature-icons">
                <div className="icon-box">⌘</div>
                <div className="icon-box">U</div>
              </div>
            </div>

            <div className="feature-card">
              <h3 className="feature-title">Collaborative Features</h3>
              <p className="feature-description">
                Share and collaborate with other musicians.
              </p>
              <span className="feature-badge">Collaborative</span>
              <div className="feature-image" style={{ height: '120px' }}>
                <Users size={40} style={{ color: 'rgba(255,255,255,0.3)' }} />
              </div>
            </div>
          </div>
        </section>

        {/* Experience Section */}
        <section className="experience-section">
          <div className="experience-grid">
            <div className="experience-content">
              <h2>Experience the Future of Music Creation</h2>
              <p>
                Our platform transforms your emotions into musical patterns, 
                enabling you to create inspiring compositions effortlessly.
              </p>
            </div>
            <div className="experience-visual">
              <div className="analysis-card">
                <div className="analysis-icon">
                  <Sparkles size={24} />
                </div>
                <div className="analysis-text">Pattern Generation</div>
              </div>
              <div className="analysis-card">
                <div className="analysis-icon">
                  <Zap size={24} />
                </div>
                <div className="analysis-text">Emotional Analysis</div>
              </div>
            </div>
          </div>
        </section>

        {/* Founder Section */}
        <section className="founder-section">
          <div className="founder-grid">
            <div className="founder-card">
              <div className="founder-image" />
              <div className="founder-info">
                <h3>Founder</h3>
                <p className="role">Abel Shegere, SWE</p>
                <span className="badge">Witnessed it first person</span>
              </div>
            </div>
            <div className="founder-content">
              <h2>By producers, for producers</h2>
              <p>
                Our user-friendly interface designed by Carnegie Mellon alumni 
                ensures a seamless experience, allowing you to focus on creativity.
              </p>
            </div>
          </div>
        </section>

        {/* Trusted Section */}
        <section className="trusted-section">
          <h2 className="trusted-title">Trusted by Leading Music Producers</h2>
          <div className="trusted-logos">
            <div className="logo-item">
              <Hash size={20} />
              LOGIC
            </div>
            <div className="logo-item">
              <Hash size={20} />
              ABLETON
            </div>
            <div className="logo-item">
              <Hash size={20} />
              FL STUDIO
            </div>
            <div className="logo-item">
              <Hash size={20} />
              KONTAKT
            </div>
            <div className="logo-item">
              <Hash size={20} />
              SERUM
            </div>
          </div>
        </section>

        {/* Testimonial */}
        <section className="testimonial-section">
          <div className="testimonial">
            <p className="testimonial-text">
              "The emotional analysis feature is truly groundbreaking and allows 
              me to spend more time editing than brainstorming sequences."
            </p>
            <div className="testimonial-author">
              <div className="author-image" />
              <div className="author-info">
                <div className="author-name">Siraj Ahmed a.k.a JARIS</div>
                <div className="author-title">Producer</div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div>© 2025 ArpeggiateAI - Powered by AI</div>
            <div className="footer-links">
              <a href="#" className="footer-link">Docs</a>
              <a href="#" className="footer-link">Blog</a>
              <a href="#" className="footer-link">Support</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
