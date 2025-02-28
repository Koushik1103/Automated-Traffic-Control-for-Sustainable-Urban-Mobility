import React from "react";
import { useNavigate } from "react-router-dom";
import "./Analytics.css";

const Analytics = () => {
  const navigate = useNavigate();
  
  return (
    <div className="analytics-container">
      <nav className="navbar">
        <button className="nav-btn" onClick={() => navigate("/home")}>Home</button>
        <button className="nav-btn" onClick={() => navigate("/admin")}>Admin</button>
      </nav>

      <div className="search-container">
        <div className="search-box">
          <input type="text" placeholder="🔍 Search" className="search-input" />
          <div className="analytics-card">
            <p>Lorem ipsum dolor sit amet.</p>
          </div>
        </div>
      </div>
    </div>
  );
};


export default Analytics;
