import React from "react";
import CandlestickChart from "./CandlestickChart";
import "./StockScanner.css";

const StockScanner = ({ stocks, loading }) => {
  console.log("📌 Stocks received from scan-stocks API:", stocks); // ✅ Debug API response

  const downloadCSV = () => {
    if (!stocks || stocks.length === 0) return;

    const csvContent = [
      ["Ticker", "Volatility", "Price Change", "RSI", "Close Price"],
      ...stocks.map((stock) => [
        stock.ticker || stock.T,  // ✅ Ensure ticker is correctly accessed
        stock.volatility ? `${(stock.volatility * 100).toFixed(2)}%` : "N/A",
        stock.price_change ? `${(stock.price_change * 100).toFixed(2)}%` : "N/A",
        stock.rsi ? stock.rsi.toFixed(2) : "N/A",
        stock.c ? stock.c.toFixed(2) : "N/A",
      ]),
    ]
      .map((row) => row.join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "scanned_stocks.csv");
    link.click();
  };

  return (
    <div className="stock-scanner">
      {/* Download Button */}
      <div style={{ textAlign: "right", marginBottom: "10px" }}>
        <button onClick={downloadCSV} className="download-button">
          Download CSV
        </button>
      </div>

      {/* Display Stock Results */}
      {loading ? (
        <p style={{ textAlign: "center" }}>Loading stocks...</p>
      ) : (
        <div className="stock-list">
          {stocks && stocks.length > 0 ? (
            stocks.map((stock) => {
              const ticker = stock.ticker || stock.T; // ✅ Ensure correct ticker
              console.log("📌 Processing stock:", stock);
              console.log("📌 Rendering ticker:", ticker);

              return (
                <div className="stock-item" key={ticker || Math.random()}>
                  {/* Statistics Section */}
                  <div className="stock-stats">
                    <h4>{ticker}</h4>
                    <p>
                      <strong>Volatility:</strong>{" "}
                      {stock.volatility
                        ? `${(stock.volatility * 100).toFixed(2)}%`
                        : "N/A"}
                    </p>
                    <p>
                      <strong>Price Change:</strong>{" "}
                      {stock.price_change
                        ? `${(stock.price_change * 100).toFixed(2)}%`
                        : "N/A"}
                    </p>
                    <p>
                      <strong>RSI:</strong> {stock.rsi ? stock.rsi.toFixed(2) : "N/A"}
                    </p>
                    <p>
                      <strong>Entry Point:</strong> $
                      {stock.c ? (stock.c * 0.95).toFixed(2) : "N/A"}
                    </p>
                    <p>
                      <strong>Exit Point:</strong> $
                      {stock.c ? (stock.c * 1.1).toFixed(2) : "N/A"}
                    </p>
                  </div>

                  {/* Chart Section */}
                  <div className="stock-chart">
                    <CandlestickChart
                      ticker={ticker}  // ✅ Pass correct ticker
                      entryPoint={stock.c ? stock.c * 0.95 : null}
                      exitPoint={stock.c ? stock.c * 1.1 : null}
                    />
                  </div>
                </div>
              );
            })
          ) : (
            <p style={{ textAlign: "center" }}>
              No stocks found. Please adjust your criteria.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default StockScanner;
