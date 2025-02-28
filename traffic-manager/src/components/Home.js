import React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="dashboard-container">
      <nav className="navbar">
        <button className="nav-btn" onClick={() => navigate("/home")}>Home</button>
        <button className="nav-btn" onClick={() => navigate("/admin")}>Admin</button>
      </nav>

      <div className="card-container">
        <div className="card one">Lorem ipsum dolor sit amet.</div>
        <div className="card clickable" onClick={() => navigate("/analytics")}>Analytics Page redirect on click</div>
      </div>
    </div>
  );
};

export default Home;
