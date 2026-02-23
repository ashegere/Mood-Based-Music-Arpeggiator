import React from 'react';
import { Sparkles, Zap } from 'lucide-react';
import './Experience.css';

const Experience = () => {
  return (
    <section className="experience-section">
      <div className="container">
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
      </div>
    </section>
  );
};

export default Experience;
