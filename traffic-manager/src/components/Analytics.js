import React from "react";
import "./Analytics.css";

const Analytics = () => {
  return (
    <div className="analytics-container">
      <nav className="navbar">
        <button className="nav-btn">Home</button>
        <button className="nav-btn">Admin</button>
      </nav>

      <div className="search-box">
        <input type="text" placeholder="🔍 Search" className="search-input" />
      </div>

      <div className="analytics-card">
        <p>Lorem ipsum dolor sit amet.</p>
      </div>
    </div>
  );
};

export default Analytics;
