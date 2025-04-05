import React, { useState, useEffect, useRef } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import StockScanner from "./components/StockScanner";
import SearchForm from "./components/SearchForm";
import TickerNewsWidget from "./components/TickerNewsWidget";
import StocksPage from "./components/StocksPage";
import OptionsPage from "./components/OptionsPage";
import CryptoPage from "./components/CryptoPage";
import ShortSalesPage from "./components/ShortSalesPage";
import NumberOnePicksPage from "./components/NumberOnePicksPage";
import "./App.css";

const App = () => {
  const [stocks, setStocks] = useState([]);
  const [tickers, setTickers] = useState([]);
  const [theme, setTheme] = useState(
    localStorage.getItem("theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef(); // ✅ Ref to the dropdown div

  useEffect(() => {
    document.body.classList.toggle("dark-mode", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  // ✅ Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const fetchStocks = async (criteria) => {
    try {
      const queryParams = new URLSearchParams({
        min_price: criteria.minPrice,
        max_price: criteria.maxPrice,
        min_rsi: criteria.minRSI,
        max_rsi: criteria.maxRSI,
        volume_surge: criteria.volumeSurge,
      });
      const response = await fetch(`http://localhost:5000/api/scan-stocks?${queryParams}`);
      const data = await response.json();
      if (data?.candidates?.length) {
        setStocks(data.candidates);
        setTickers(data.candidates.map((stock) => stock.T));
      } else {
        setStocks([]);
        setTickers([]);
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setStocks([]);
      setTickers([]);
    }
  };

  return (
    <Router>
      <div className={`menu-bar ${theme}`}>
        <h1 className="menu-title">AI Stock Scanner</h1>
        <div className="menu-buttons">
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </button>
          <Link to="/"><button>Home</button></Link>

          {/* ✅ Dropdown wrapper with ref */}
          <div
  className="dropdown"
  onMouseEnter={() => setShowDropdown(true)}
  onMouseLeave={() => setShowDropdown(false)}
>
  <button className="dropdown-btn">
    Stocks ▾
  </button>

  {showDropdown && (
    <div className="dropdown-content">
      <Link to="/stocks" onClick={() => setShowDropdown(false)}>All Stocks</Link>
      <Link to="/number-one-picks" onClick={() => setShowDropdown(false)}>Number One Picks</Link>
    </div>
  )}
</div>


          <Link to="/options"><button>Options</button></Link>
          <Link to="/crypto"><button>Crypto</button></Link>
          <Link to="/short-sales"><button>Short Sales</button></Link>
        </div>
      </div>

      <Routes>
        <Route
          path="/"
          element={
            <div className={`app-layout ${theme}`}>
              <div className="search-bar"><SearchForm onSearch={fetchStocks} /></div>
              <div className="stock-results-header"><h2>Stock Results</h2></div>
              <div className="main-content">
                <StockScanner stocks={stocks} />
                <div className="news-widget"><TickerNewsWidget tickers={tickers} /></div>
              </div>
            </div>
          }
        />
        <Route path="/stocks" element={<StocksPage />} />
        <Route path="/number-one-picks" element={<NumberOnePicksPage />} />
        <Route path="/options" element={<OptionsPage />} />
        <Route path="/crypto" element={<CryptoPage />} />
        <Route path="/short-sales" element={<ShortSalesPage />} />
      </Routes>
    </Router>
  );
};

export default App;
