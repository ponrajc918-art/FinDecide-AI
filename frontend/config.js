window.CONFIG = {
  API_URL:
    window.FINDECIDE_API_URL ||
    (window.location.hostname === "localhost" ||
     window.location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://findecideai.onrender.com")
};
