import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

const Home = () => {
  const navigate = useNavigate();
  const [images, setImages] = useState({ direction_1: "", direction_2: "" });
  const [trafficSignal, setTrafficSignal] = useState({
    direction_1: "",
    direction_2: "",
    pedestrian: "",
  });

  useEffect(() => {
    fetch("http://localhost:5000/get_traffic_data")
      .then((response) => response.json())
      .then((data) => {
        setImages(data.latest_frames);
        setTrafficSignal(data.traffic_signal);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  return (
    <div className="dashboard-container">
      <nav className="navbar">
        <button className="nav-btn" onClick={() => navigate("/home")}>
          Home
        </button>
        <button className="nav-btn" onClick={() => navigate("/admin")}>
          Admin
        </button>
        <button className="nav-btn" onClick={() => navigate("/analytics")}>
          Analytics
        </button>
      </nav>

      <div className="card-container">
        <div className="card-container-two">
          <div className="card one">
            <p>Signal: {trafficSignal.direction_1}</p>
            <img
              src={`data:image/png;base64,${images.direction_1}`}
              alt="Traffic 1"
              className="traffic-image"
            />
          </div>
          <div className="card two">
            <p>Signal: {trafficSignal.direction_2}</p>
            <img
              src={`data:image/png;base64,${images.direction_2}`}
              alt="Traffic 2"
              className="traffic-image"
            />
          </div>
        </div>
        {/* <div className="card clickable" onClick={() => navigate("/analytics")}>Analytics Page redirect on click</div> */}

        <div className="card three">
          <p>Pedestrian Signal: {trafficSignal.pedestrian}</p>
        </div>
      </div>
    </div>
  );
};

export default Home;

/*
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
*/
