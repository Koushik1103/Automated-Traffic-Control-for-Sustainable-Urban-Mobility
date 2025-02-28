import React from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";

const Login = () => {
    const navigate = useNavigate();
    
    return (
        <div className="login-container">
            <div className="split-login">
                <div className="login-box">
                    <input type="email" placeholder="Email" className="input-field top" />
                    <input type="password" placeholder="Password" className="input-field" />
                    <button className="login-btn" onClick={() => navigate("/home")}>Log in</button>
                    <div className="forgot-password" onClick={() => navigate("/")}>Forgot Password?</div>
                </div>
            </div>
        </div>
    );
};
    
export default Login;