import React from "react";
import CandlestickChart from "./CandlestickChart";
import "./StockScanner.css";

const StockScanner = ({ stocks, loading }) => {
  // 🟢 DEBUG: Log API response to ensure data contains entry & exit points
  console.log("📌 DEBUG: Received stocks data", stocks);

  const downloadCSV = () => {
    if (!stocks || stocks.length === 0) return;

    const csvContent = [
      ["Ticker", "Volatility", "Price Change", "RSI", "Close Price", "Entry Point", "Exit Point"],
      ...stocks.map((stock) => [
        stock.T,
        stock.volatility ? `${(stock.volatility * 100).toFixed(2)}%` : "N/A",
        stock.price_change ? `${(stock.price_change * 100).toFixed(2)}%` : "N/A",
        stock.rsi ? stock.rsi.toFixed(2) : "N/A",
        stock.c ? stock.c.toFixed(2) : "N/A",
        stock.entry_point ? `$${parseFloat(stock.entry_point).toFixed(2)}` : "N/A",
        stock.exit_point ? `$${parseFloat(stock.exit_point).toFixed(2)}` : "N/A",
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
              // 🟢 DEBUG: Log each stock's entry and exit points
              console.log(
                `📌 DEBUG: Rendering stock - ${stock.T}, Entry: ${stock.entry_point}, Exit: ${stock.exit_point}`
              );

              // ✅ Ensure entry & exit are converted to numbers and handle NaN cases
              const entryPrice =
                stock.entry_point && !isNaN(stock.entry_point) ? parseFloat(stock.entry_point) : "N/A";
              const exitPrice =
                stock.exit_point && !isNaN(stock.exit_point) ? parseFloat(stock.exit_point) : "N/A";

              // 🟢 DEBUG: Log values before passing to CandlestickChart
              console.log(
                `📌 DEBUG: Passing to CandlestickChart - ${stock.T}, Entry: ${entryPrice}, Exit: ${exitPrice}`
              );

              return (
                <div className="stock-item" key={stock.T}>
                  {/* Statistics Section */}
                  <div className="stock-stats">
                    <h4>{stock.T}</h4>
                    <p>
                      <strong>Volatility:</strong>{" "}
                      {stock.volatility ? `${(stock.volatility * 100).toFixed(2)}%` : "N/A"}
                    </p>
                    <p>
                      <strong>Price Change:</strong>{" "}
                      {stock.price_change ? `${(stock.price_change * 100).toFixed(2)}%` : "N/A"}
                    </p>
                    <p>
                      <strong>RSI:</strong> {stock.rsi ? stock.rsi.toFixed(2) : "N/A"}
                    </p>
                    <p>
                      <strong>Entry Point:</strong>{" "}
                      {entryPrice !== "N/A" ? `$${entryPrice.toFixed(2)}` : "N/A"}
                    </p>
                    <p>
                      <strong>Exit Point:</strong>{" "}
                      {exitPrice !== "N/A" ? `$${exitPrice.toFixed(2)}` : "N/A"}
                    </p>
                  </div>

                  {/* Chart Section */}
                  <div className="stock-chart">
                    <CandlestickChart
                      ticker={stock.T}
                      entryPoint={entryPrice !== "N/A" ? entryPrice : 0}
                      exitPoint={exitPrice !== "N/A" ? exitPrice : 0}
                    />
                  </div>
                </div>
              );
            })
          ) : (
            <p style={{ textAlign: "center" }}>No stocks found. Please adjust your criteria.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default StockScanner;
