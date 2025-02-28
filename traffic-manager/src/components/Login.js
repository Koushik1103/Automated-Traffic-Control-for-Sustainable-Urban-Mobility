import React from "react";
import "./Login.css";

const Login = () => {
    return (
        <div className="login-container">
            <div className="split-login">
                <div className="login-box">
                    <input type="email" placeholder="Email" className="input-field top" />
                    <input type="password" placeholder="Password" className="input-field" />
                    <button className="login-btn">Log in</button>
                    <a href="/" className="forgot-password">Forgot Password?</a>
                </div>
            </div>
        </div>
    );
};
    
export default Login;