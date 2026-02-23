import React, { useState, useEffect } from 'react';
import { Music, Mail, Lock, Eye, EyeOff } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import './Login.css';

const Login = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  useEffect(() => {
    // Redirect to home if user is already logged in
    const token = localStorage.getItem('token');
    if (token) {
      navigate('/home');
    }
  }, [navigate]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError(''); // Clear error on input change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8006/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      // Store token
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user', JSON.stringify(data.user));

      // Redirect to home page
      navigate('/home');
    } catch (err) {
      setError(err.message || 'An error occurred during login');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    const width = 500;
    const height = 600;
    const left = window.screen.width / 2 - width / 2;
    const top = window.screen.height / 2 - height / 2;

    const popup = window.open(
      'http://localhost:8006/api/auth/google/login',
      'Google Sign In',
      `width=${width},height=${height},left=${left},top=${top}`
    );

    // Listen for messages from the popup
    window.addEventListener('message', (event) => {
      if (event.origin !== 'http://localhost:8006') return;

      if (event.data.type === 'GOOGLE_AUTH_SUCCESS') {
        // Store token and user data
        localStorage.setItem('token', event.data.access_token);
        localStorage.setItem('user', JSON.stringify(event.data.user));

        // Close popup if still open
        if (popup && !popup.closed) {
          popup.close();
        }

        // Redirect to home page
        navigate('/home');
      } else if (event.data.type === 'GOOGLE_AUTH_ERROR') {
        setError(event.data.error || 'Google authentication failed');
        if (popup && !popup.closed) {
          popup.close();
        }
      }
    });
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <Link to="/" className="logo">
            <div className="logo-icon">
              <Music size={24} strokeWidth={3} />
            </div>
            <span>arpeggiator.ai</span>
          </Link>
          <h1>Welcome Back</h1>
          <p>Sign in to your account to continue creating music</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <div className="input-wrapper">
              <Mail size={20} className="input-icon" />
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="Enter your email"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="input-wrapper">
              <Lock size={20} className="input-icon" />
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Enter your password"
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          <div className="form-options">
            <label className="checkbox-label">
              <input type="checkbox" />
              <span className="checkmark"></span>
              Remember me
            </label>
            <a href="#" className="forgot-password">Forgot password?</a>
          </div>

          <button type="submit" className="login-btn" disabled={loading}>
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <div className="divider">
          <span>or</span>
        </div>

        <button className="google-btn" onClick={handleGoogleLogin}>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19.8055 10.2292C19.8055 9.55156 19.7501 8.86719 19.6323 8.19531H10.2002V12.0492H15.6014C15.3776 13.2911 14.6568 14.3898 13.6034 15.0875V17.5867H16.825C18.7171 15.8449 19.8055 13.2719 19.8055 10.2292Z" fill="#4285F4"/>
            <path d="M10.2002 20C12.9502 20 15.2683 19.1042 16.8319 17.5867L13.6103 15.0875C12.7145 15.6958 11.5517 16.0428 10.2071 16.0428C7.5464 16.0428 5.28546 14.2803 4.4965 11.9167H1.17188V14.4921C2.76854 17.6594 6.31437 20 10.2002 20Z" fill="#34A853"/>
            <path d="M4.48963 11.9167C4.0173 10.6748 4.0173 9.32943 4.48963 8.0875V5.51208H1.17188C-0.393959 8.63583 -0.393959 12.3683 1.17188 15.4921L4.48963 11.9167Z" fill="#FBBC04"/>
            <path d="M10.2002 3.95729C11.6239 3.93646 13.0029 4.47396 14.0424 5.45833L16.8874 2.61333C15.1801 0.990625 12.9294 0.0739583 10.2002 0.100625C6.31437 0.100625 2.76854 2.44062 1.17188 5.51187L4.48963 8.08729C5.27171 5.71687 7.5395 3.95729 10.2002 3.95729Z" fill="#EA4335"/>
          </svg>
          Continue with Google
        </button>

        <div className="login-footer">
          <p>Don't have an account? <Link to="/signup" className="signup-link">Sign up</Link></p>
        </div>
      </div>
    </div>
  );
};

export default Login;
