import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { login as apiLogin } from '../services/api';
import imgLogo from '../assets/logo.png';

const Login = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({ email: '', password: '' });

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) navigate('/build');
  }, [navigate]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const data = await apiLogin(formData.email, formData.password);
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user', JSON.stringify(data.user));
      navigate('/build');
    } catch (err) {
      setError(err.message || 'An error occurred during login');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    const width = 500, height = 600;
    const left = window.screen.width / 2 - width / 2;
    const top = window.screen.height / 2 - height / 2;
    const popup = window.open(
      '/api/auth/google/login',
      'Google Sign In',
      `width=${width},height=${height},left=${left},top=${top}`
    );
    window.addEventListener('message', (event) => {
      if (event.origin !== 'http://localhost:8006') return;
      if (event.data.type === 'GOOGLE_AUTH_SUCCESS') {
        localStorage.setItem('token', event.data.access_token);
        localStorage.setItem('user', JSON.stringify(event.data.user));
        if (popup && !popup.closed) popup.close();
        navigate('/build');
      } else if (event.data.type === 'GOOGLE_AUTH_ERROR') {
        setError(event.data.error || 'Google authentication failed');
        if (popup && !popup.closed) popup.close();
      }
    });
  };

  return (
    <div className="login-root">
      <style>{`
        @import url('https://api.fontshare.com/v2/css?f[]=clash-grotesk@200,300,400,500,600,700&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        .login-root {
          font-family: 'ClashGrotesk-Variable', 'Clash Grotesk', sans-serif;
          background: #0000ff;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          padding: 12px 24px 24px;
          gap: 0;
          overflow-x: hidden;
        }

        /* ── NAVBAR ── */
        .login-navbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
          position: relative;
          height: 64px;
          flex-shrink: 0;
        }

        .login-logo {
          display: flex;
          align-items: center;
          cursor: pointer;
          text-decoration: none;
        }
        .login-logo img { height: 36px; width: auto; object-fit: contain; display: block; }

        .login-nav-links {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 40px;
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(245,245,245,0.2);
          border-radius: 40px;
          padding: 13px 28px;
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
        }

        .login-nav-link {
          font-size: 15px;
          font-weight: 500;
          color: #ffffff;
          text-decoration: none;
          white-space: nowrap;
          transition: opacity 0.15s;
        }

        .login-nav-link:hover { opacity: 0.75; }

        /* ── MAIN AREA ── */
        .login-main {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 48px 0 32px;
        }

        /* ── CARD ── */
        .login-card {
          background: #0000ff;
          border: 2px solid #ffffff;
          border-radius: 13px;
          box-shadow: 0 4px 4px rgba(0,0,0,0.25);
          padding: 38px 35px;
          width: 100%;
          max-width: 480px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0;
        }

        .login-heading {
          font-size: clamp(22px, 3vw, 34px);
          font-weight: 400;
          color: #ffffff;
          text-align: center;
          letter-spacing: -0.9px;
          margin-bottom: 26px;
        }

        /* ── FORM ── */
        .login-form {
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 13px;
        }

        .login-input {
          width: 100%;
          height: 51px;
          background: #ffffff;
          border: none;
          border-radius: 40px;
          padding: 0 26px;
          font-size: 14px;
          font-weight: 500;
          font-family: inherit;
          color: #060511;
          outline: none;
          transition: box-shadow 0.15s;
        }

        .login-input::placeholder {
          color: rgba(6,5,17,0.4);
        }

        .login-input:focus {
          box-shadow: 0 0 0 2px #bfff3e;
        }

        .login-input-wrap {
          position: relative;
          width: 100%;
        }

        .login-password-toggle {
          position: absolute;
          right: 19px;
          top: 50%;
          transform: translateY(-50%);
          background: none;
          border: none;
          cursor: pointer;
          color: rgba(6,5,17,0.4);
          display: flex;
          align-items: center;
          padding: 0;
          font-size: 14px;
        }

        .login-error {
          background: rgba(255,80,80,0.15);
          border: 1px solid rgba(255,80,80,0.4);
          border-radius: 10px;
          padding: 8px 13px;
          font-size: 11px;
          color: #ff8080;
          text-align: center;
          width: 100%;
        }

        .login-submit-btn {
          width: 100%;
          height: 51px;
          background: #bfff3e;
          border: none;
          border-radius: 40px;
          font-size: 19px;
          font-weight: 500;
          font-family: inherit;
          color: #060511;
          cursor: pointer;
          transition: opacity 0.15s;
          margin-top: 6px;
        }

        .login-submit-btn:hover { opacity: 0.88; }
        .login-submit-btn:disabled { opacity: 0.6; cursor: not-allowed; }

        /* ── DIVIDER ── */
        .login-divider {
          display: flex;
          align-items: center;
          gap: 13px;
          width: 100%;
          margin: 16px 0;
        }

        .login-divider-line {
          flex: 1;
          height: 1px;
          background: rgba(255,255,255,0.35);
        }

        .login-divider-text {
          font-size: 18px;
          font-weight: 500;
          color: #ffffff;
          white-space: nowrap;
        }

        /* ── GOOGLE BUTTON ── */
        .login-google-btn {
          width: 100%;
          height: 51px;
          background: #ffffff;
          border: none;
          border-radius: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 13px;
          font-size: 18px;
          font-weight: 500;
          font-family: inherit;
          color: #060511;
          cursor: pointer;
          transition: opacity 0.15s;
        }

        .login-google-btn:hover { opacity: 0.88; }

        .login-google-icon {
          width: 19px;
          height: 19px;
          flex-shrink: 0;
        }

        /* ── SIGN UP LINK ── */
        .login-signup-row {
          margin-top: 19px;
          font-size: 13px;
          font-weight: 500;
          color: #ffffff;
          text-align: center;
        }

        .login-signup-row a {
          color: #ffffff;
          text-decoration: underline;
          text-underline-offset: 3px;
          text-decoration-thickness: 1px;
          transition: opacity 0.15s;
        }

        .login-signup-row a:hover { opacity: 0.75; }

        /* ── FOOTER ── */
        .login-footer {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 19px;
          border-top: 1px solid rgba(255,255,255,0.2);
          margin-top: auto;
          flex-shrink: 0;
        }

        .login-footer-copy {
          font-size: 11px;
          font-weight: 500;
          color: rgba(255,255,255,0.6);
          white-space: nowrap;
        }

        .login-footer-coffee {
          background: rgba(255,255,255,0.1);
          border: none;
          border-radius: 8px;
          padding: 6px 16px;
          height: 32px;
          font-size: 11px;
          font-weight: 500;
          font-family: inherit;
          color: #ffffff;
          cursor: pointer;
          text-decoration: none;
          display: flex;
          align-items: center;
          transition: opacity 0.15s;
          white-space: nowrap;
        }

        .login-footer-coffee:hover { opacity: 0.75; }

        /* ── ANIMATIONS ── */
        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-24px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .login-navbar {
          animation: fadeInDown 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        .login-card {
          animation: fadeInDown 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.15s both;
        }

        .login-heading {
          animation: fadeInDown 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.3s both;
        }

        .login-form .login-input:nth-of-type(1),
        .login-form .login-input-wrap:nth-of-type(1) {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.4s both;
        }

        .login-form .login-input-wrap {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.5s both;
        }

        .login-submit-btn {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.6s both;
        }

        .login-divider {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.7s both;
        }

        .login-google-btn {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.8s both;
        }

        .login-signup-row {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.9s both;
        }

        .login-footer {
          animation: fadeInDown 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.5s both;
        }

        /* ── RESPONSIVE ── */
        @media (max-width: 768px) {
          .login-root { padding: 13px; }
          .login-nav-links { display: none; }
          .login-card { padding: 29px 22px; }
          .login-heading { font-size: 22px; }
          .login-input { height: 45px; font-size: 13px; padding: 0 19px; }
          .login-submit-btn { height: 45px; font-size: 16px; }
          .login-google-btn { height: 45px; font-size: 14px; }
          .login-footer { flex-direction: column; gap: 10px; align-items: flex-start; }
        }

        @media (max-width: 480px) {
          .login-root { padding: 10px; }
          .login-card { padding: 22px 16px; border-radius: 10px; }
          .login-input { height: 42px; padding: 0 16px; }
          .login-submit-btn { height: 42px; }
          .login-google-btn { height: 42px; font-size: 13px; gap: 10px; }
          .login-divider-text { font-size: 14px; }
        }
      `}</style>

      {/* ── NAVBAR ── */}
      <nav className="login-navbar">
        <Link to="/" className="login-logo" aria-label="Arpeggiate home">
          <img src={imgLogo} alt="Arpeggiate" />
        </Link>

        <div className="login-nav-links">
          <Link to="/" className="login-nav-link">Home</Link>
          <a href="/#create" className="login-nav-link">Explore</a>
          <a href="/#create" className="login-nav-link">Create</a>
        </div>

        {/* spacer to balance logo */}
        <div style={{ width: '34px' }} />
      </nav>

      {/* ── CARD ── */}
      <main className="login-main">
        <div className="login-card">
          <img src={imgLogo} alt="Arpeggiate" style={{ height: '40px', width: 'auto', objectFit: 'contain', display: 'block', margin: '0 auto 16px' }} />
          <h1 className="login-heading">Sign in to <span style={{ color: '#d9ff00' }}>Arpeggiate</span></h1>

          <form className="login-form" onSubmit={handleSubmit}>
            <input
              className="login-input"
              type="email"
              name="email"
              placeholder="Email address"
              value={formData.email}
              onChange={handleInputChange}
              required
              autoComplete="email"
            />

            <div className="login-input-wrap">
              <input
                className="login-input"
                type={showPassword ? 'text' : 'password'}
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleInputChange}
                required
                autoComplete="current-password"
              />
              <button
                type="button"
                className="login-password-toggle"
                onClick={() => setShowPassword(v => !v)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? '🙈' : '👁'}
              </button>
            </div>

            {error && <div className="login-error">{error}</div>}

            <button type="submit" className="login-submit-btn" disabled={loading}>
              {loading ? 'Signing In…' : 'Sign In'}
            </button>
          </form>

          <div className="login-divider">
            <div className="login-divider-line" />
            <span className="login-divider-text">or</span>
            <div className="login-divider-line" />
          </div>

          <button className="login-google-btn" onClick={handleGoogleLogin}>
            <svg className="login-google-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </button>

          <p className="login-signup-row">
            Don't have an account? <Link to="/signup">Sign Up</Link>
          </p>
        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer className="login-footer">
        <span className="login-footer-copy">© 2025 Arpeggiate.ai - Powered by AI</span>
        <a
          href="https://buymeacoffee.com"
          target="_blank"
          rel="noopener noreferrer"
          className="login-footer-coffee"
        >
          Buy me a coffee
        </a>
      </footer>
    </div>
  );
};

export default Login;
