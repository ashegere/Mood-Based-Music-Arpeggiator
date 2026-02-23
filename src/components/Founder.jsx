import React from 'react';
import './Founder.css';

const Founder = () => {
  return (
    <section className="founder-section">
      <div className="container">
        <div className="founder-grid">
          <div className="founder-card">
            <div className="founder-image" />
            <div className="founder-info">
              <h3>Founder</h3>
              <p className="role">Abel Shiferaw, SWE</p>
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
      </div>
    </section>
  );
};

export default Founder;
