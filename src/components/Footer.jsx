import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div>© 2025 ArpeggiateAI - Powered by AI</div>
          <div className="footer-links">
            <a href="#docs" className="footer-link">Docs</a>
            <a href="#blog" className="footer-link">Blog</a>
            <a href="#support" className="footer-link">Support</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
