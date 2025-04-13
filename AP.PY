import streamlit as st

###############################################################################
# PAGE CONFIG & LAYOUT
###############################################################################
st.set_page_config(page_title="Login with Image Demo", layout="wide")

###############################################################################
# SESSION STATE FOR LOGIN
###############################################################################
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

###############################################################################
# ADMIN CREDENTIALS (SIMPLE DEMO)
###############################################################################
ADMIN_USERNAME = "Administrator"
ADMIN_PASSWORD = "BWB2025"

###############################################################################
# CSS FOR A COMPACT LOGIN PAGE & IMAGE
###############################################################################
st.markdown("""
<style>
/* Hide Streamlit default UI elements (optional) */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding: 0 !important;
    margin: 0 !important;
}

/* Full-page background & centering the login container */
body {
    background-color: #f2f2f2;
    margin: 0; 
    padding: 0; 
    font-family: 'Georgia', serif;
    color: #333;
}
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh; /* fill entire screen height */
    padding: 1rem;
}
/* The login box */
.login-box {
    background-color: #fff;
    padding: 1.5rem 2rem;
    border-radius: 8px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    max-width: 320px;
    width: 100%;
    text-align: center;
    font-family: 'Georgia', serif;
}
.login-box img {
    margin-bottom: 1rem;
    max-width: 100px;  /* Adjust as needed for your image size */
}
.login-box h2 {
    margin-bottom: 1rem;
    font-size: 1.2rem;
    color: #444;
}
/* Smaller text inputs */
.login-box input {
    width: 100%;
    padding: 0.5rem;
    margin-bottom: 0.8rem;
    border-radius: 4px;
    border: 1px solid #ccc;
    font-size: 0.9rem;
}
/* Login button styling */
.login-box button {
    width: 100%;
    padding: 0.5rem;
    font-size: 0.9rem;
    color: #fff;
    background-color: #007bff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.login-box button:hover {
    background-color: #0069d9;
}
</style>
""", unsafe_allow_html=True)

###############################################################################
# LOGIN LOGIC
###############################################################################
if not st.session_state.logged_in:
    # Center container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    # Box with white background
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    # 1) The image: Replace with your actual file path or URL
    st.markdown('<img src="path/to/your_image.png" alt="Logo">', unsafe_allow_html=True)

    # 2) Heading
    st.markdown("<h2>Login</h2>", unsafe_allow_html=True)
    
    # 3) Login Fields
    username_input = st.text_input("Username", placeholder="Enter username")
    password_input = st.text_input("Password", placeholder="Enter password", type="password")

    # 4) Login Button
    if st.button("Login"):
        if username_input == ADMIN_USERNAME and password_input == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)  # close .login-box
    st.markdown("</div>", unsafe_allow_html=True)  # close .login-container
    st.stop()

###############################################################################
# AFTER LOGIN: SHOW DASHBOARD
###############################################################################
st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()

st.title("Main Dashboard")
st.write("Add your main dashboard content here.")
