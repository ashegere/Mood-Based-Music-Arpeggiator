import React from 'react';
import './Testimonial.css';

const Testimonial = () => {
  return (
    <section className="testimonial-section">
      <div className="container">
        <div className="testimonial">
          <p className="testimonial-text">
            "The emotional analysis feature is truly groundbreaking and allows 
            me to spend more time editing than brainstorming sequences."
          </p>
          <div className="testimonial-author">
            <div className="author-image" />
            <div className="author-info">
              <div className="author-name">Jane Doyle</div>
              <div className="author-title">Producer</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonial;
