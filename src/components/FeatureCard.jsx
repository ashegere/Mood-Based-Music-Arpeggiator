import React from 'react';
import './FeatureCard.css';

const FeatureCard = ({ 
  title, 
  description, 
  icon, 
  type, 
  hasImage, 
  badge, 
  icons, 
  gradient,
  imageHeight 
}) => {
  return (
    <div className={`feature-card ${type === 'large' ? 'large' : ''}`}>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-description">{description}</p>
      
      {badge && <span className="feature-badge">{badge}</span>}
      
      {type === 'icons' && icons && (
        <div className="feature-icons">
          {icons.map((iconText, index) => (
            <div key={index} className="icon-box">{iconText}</div>
          ))}
        </div>
      )}
      
      {hasImage && (
        <div 
          className="feature-image" 
          style={{ 
            height: imageHeight || '180px',
            background: gradient 
              ? 'linear-gradient(135deg, rgba(255,107,0,0.1) 0%, rgba(99,102,241,0.1) 100%)'
              : 'rgba(0, 0, 0, 0.3)'
          }}
        >
          {icon && (
            <div style={{ textAlign: 'center' }}>
              {icon}
              {type === 'large' && title === 'Emotion-driven Patterns' && (
                <div style={{ 
                  fontSize: '16px', 
                  fontWeight: 600, 
                  color: 'rgba(255,255,255,0.7)', 
                  marginTop: '12px' 
                }}>
                  Emotion Analysis
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FeatureCard;
