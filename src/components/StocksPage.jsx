import React, { useState, useEffect } from "react";
import ErrorBoundary from "./ErrorBoundary";
import StockTracker from "./StockTracker"; // Live Stock Tracker Component
import io from "socket.io-client";
import "./StocksPage.css";

import {
  StockChartComponent,
  StockChartSeriesCollectionDirective,
  StockChartSeriesDirective,
  StockChartIndicatorsDirective,
  StockChartIndicatorDirective,
  Inject,
  DateTime,
  Tooltip,
  RangeTooltip,
  Crosshair,
  LineSeries,
  CandleSeries,
  Legend,
  Export,
  EmaIndicator,
  TmaIndicator,
  SmaIndicator,
  MomentumIndicator,
  AtrIndicator,
  AccumulationDistributionIndicator,
  BollingerBands,
  MacdIndicator,
  StochasticIndicator,
  RsiIndicator,
  AnnotationsDirective,
  AnnotationDirective,
} from "@syncfusion/ej2-react-charts";

// Initialize WebSocket connection
const socket = io("http://localhost:5000");

const StocksPage = () => {
  const [ticker, setTicker] = useState("");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedIndicator, setSelectedIndicator] = useState("BollingerBands");
  const [recommendations, setRecommendations] = useState([]);

  const periods = [
    { intervalType: "Months", interval: 1, text: "1M" },
    { intervalType: "Months", interval: 3, text: "3M" },
    { intervalType: "Months", interval: 6, text: "6M" },
    { intervalType: "Years", interval: 1, text: "YTD" },
    { intervalType: "Years", interval: 3, text: "All" },
  ];

  // Fetch historical chart data
  const fetchChartData = async (tickerSymbol) => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:5000/api/candlestick?ticker=${tickerSymbol}`
      );
      const data = await response.json();
      if (data.dates) {
        const formattedData = data.dates.map((date, index) => ({
          x: new Date(date),
          open: data.open[index],
          high: data.high[index],
          low: data.low[index],
          close: data.close[index],
        }));
        setChartData(formattedData);
        setError("");
      } else {
        setChartData([]);
        setError("No data available for the selected ticker.");
      }
    } catch (err) {
      console.error("Error fetching chart data:", err);
      setChartData([]);
      setError("Failed to load chart data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChartData(selectedTicker);
  }, [selectedTicker]);

  // WebSocket Listener for Real-Time Updates
  useEffect(() => {
    socket.emit("track_stock", { ticker: selectedTicker });

    socket.on("stock_update", (data) => {
      console.log("Real-time data:", data);

      // Update chart data with live price
      setChartData((prevData) => [
        ...prevData,
        { x: new Date(data.timestamp), open: data.price, high: data.price, low: data.price, close: data.price },
      ]);

      // Update recommendations
      setRecommendations((prev) => [
        ...prev,
        { timestamp: data.timestamp, recommendation: data.recommendation },
      ]);
    });

    return () => {
      socket.disconnect();
    };
  }, [selectedTicker]);

  const handleSearch = () => {
    if (ticker.trim() !== "") {
      setSelectedTicker(ticker.toUpperCase());
    }
  };

  return (
    <div className="stocks-page" style={{ padding: "20px" }}>
      <div className="stock-header" style={{ marginBottom: "20px" }}>
        <h2 style={{ textAlign: "center", fontSize: "24px" }}>
          {selectedTicker} Stock Analysis
        </h2>
      </div>

      <div className="realtime-stock-tracker" style={{ marginBottom: "20px" }}>
        <h3 style={{ textAlign: "center" }}>Real-Time Stock Tracker</h3>
        <StockTracker ticker={selectedTicker} />
      </div>

      <div
        className="search-stock"
        style={{
          marginBottom: "20px",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <input
          type="text"
          placeholder="Enter stock ticker (e.g., AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          style={{
            padding: "10px",
            fontSize: "16px",
            borderRadius: "5px",
            border: "1px solid #ccc",
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            borderRadius: "5px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            cursor: "pointer",
          }}
        >
          Search
        </button>
      </div>

      <div
        id="chart-container"
        style={{
          width: "100%",
          height: "600px",
          backgroundColor: "#fff",
          borderRadius: "8px",
          padding: "10px",
        }}
      >
        {loading ? (
          <p style={{ textAlign: "center", color: "gray" }}>Loading...</p>
        ) : error ? (
          <p style={{ color: "red", textAlign: "center" }}>{error}</p>
        ) : (
          <ErrorBoundary>
            <StockChartComponent
              id="stockchart"
              enableSelector={true}
              primaryXAxis={{
                valueType: "DateTime",
                labelFormat: "MMM dd",
                majorGridLines: { width: 0 },
                intervalType: "Days",
                crosshairTooltip: { enable: true },
              }}
              primaryYAxis={{
                labelFormat: "${value}",
                majorGridLines: { width: 0 },
                rangePadding: "None",
                crosshairTooltip: { enable: true },
              }}
              tooltip={{ enable: true }}
              crosshair={{ enable: true }}
              periods={periods}
              title={`${selectedTicker} Stock Analysis`}
              height="100%"
              width="100%"
            >
              <Inject
                services={[
                  DateTime,
                  Tooltip,
                  RangeTooltip,
                  Crosshair,
                  LineSeries,
                  CandleSeries,
                  Legend,
                  Export,
                  EmaIndicator,
                  TmaIndicator,
                  SmaIndicator,
                  MomentumIndicator,
                  AtrIndicator,
                  AccumulationDistributionIndicator,
                  BollingerBands,
                  MacdIndicator,
                  StochasticIndicator,
                  RsiIndicator,
                ]}
              />
              <StockChartSeriesCollectionDirective>
                <StockChartSeriesDirective
                  dataSource={chartData}
                  xName="x"
                  open="open"
                  high="high"
                  low="low"
                  close="close"
                  type="Candle"
                  animation={{ enable: true }}
                />
              </StockChartSeriesCollectionDirective>
              <AnnotationsDirective>
                {recommendations.map((rec, index) => (
                  <AnnotationDirective
                    key={index}
                    content={`<div>${rec.recommendation}</div>`}
                    x={new Date(rec.timestamp)}
                    y={chartData[chartData.length - 1]?.close || 0}
                  />
                ))}
              </AnnotationsDirective>
            </StockChartComponent>
          </ErrorBoundary>
        )}
      </div>
    </div>
  );
};

export default StocksPage;
