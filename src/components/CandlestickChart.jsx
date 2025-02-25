import React, { useEffect, useState } from "react";
import {
  ChartComponent,
  SeriesCollectionDirective,
  SeriesDirective,
  Inject,
  DateTime,
  CandleSeries,
  Tooltip,
  Zoom,
  Crosshair,
  Legend,
} from "@syncfusion/ej2-react-charts";
import { registerLicense } from "@syncfusion/ej2-base";

// Register the Syncfusion license
registerLicense("Ngo9BigBOggjHTQxAR8/V1NMaF5cXmRCf1FpRmJGdld5fUVHYVZUTXxaS00DNHVRdkdmWX5ednVWQ2BfVEJ+WEY=");

const CandlestickChart = ({ ticker, entryPoint, exitPoint, additionalData, pageType }) => {
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState("");
  const [darkMode, setDarkMode] = useState(
    window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
  );
  useEffect(() => {
    console.log(`📌 CandlestickChart received ticker:`, ticker);
  }, [ticker]);
  

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/candlestick?ticker=${ticker}`);
        const data = await response.json();

        if (data.dates && Array.isArray(data.dates) && data.dates.length > 0) {
          // Get today's date to filter out future dates
          const today = new Date().toISOString().split("T")[0];

          // Filter out future dates
          const filteredDates = data.dates.filter((date) => date <= today);
          const indexMap = data.dates
            .map((date, index) => (date <= today ? index : null))
            .filter((x) => x !== null);

          const formattedData = indexMap.map((i) => ({
            x: new Date(data.dates[i]),
            open: data.open[i],
            high: data.high[i],
            low: data.low[i],
            close: data.close[i],
          }));

          setChartData(formattedData);
          setError(""); // Reset error if successful
        } else {
          setError("No valid historical data available for this stock.");
        }
      } catch (err) {
        console.error(`Error fetching data for ${ticker}:`, err);
        setError("Failed to load chart data.");
      }
    };

    fetchChartData();
  }, [ticker]);

  // Detect system dark mode changes
  useEffect(() => {
    const darkModeListener = window.matchMedia("(prefers-color-scheme: dark)");
    const handleDarkModeChange = (e) => setDarkMode(e.matches);
    darkModeListener.addEventListener("change", handleDarkModeChange);

    return () => darkModeListener.removeEventListener("change", handleDarkModeChange);
  }, []);

  // Chart size based on page type
  const chartHeight = pageType === "stocksPage" ? "100%" : "400px";
  const chartWidth = pageType === "stocksPage" ? "100%" : "100%";

  return (
    <div className={`chart-container ${darkMode ? "dark-mode" : ""}`}>
      {error ? (
        <p style={{ color: "red", textAlign: "center" }}>{error}</p>
      ) : (
        <ChartComponent
          id={`chart-${ticker}`}
          primaryXAxis={{ valueType: "DateTime", labelFormat: "MMM dd", intervalType: "Days" }}
          primaryYAxis={{ labelFormat: "${value}" }}
          tooltip={{ enable: true }}
          crosshair={{ enable: true, lineType: "Both" }}
          zoomSettings={{ enableMouseWheelZooming: true, mode: "XY" }}
          height={chartHeight}
          width={chartWidth}
          legendSettings={{ visible: true }}
          background={darkMode ? "#121212" : "#ffffff"} // ✅ Dark mode background
        >
          <Inject services={[CandleSeries, DateTime, Tooltip, Zoom, Crosshair, Legend]} />
          <SeriesCollectionDirective>
            <SeriesDirective
              dataSource={chartData}
              xName="x"
              open="open"
              high="high"
              low="low"
              close="close"
              type="Candle"
              name={ticker}
              animation={{ enable: true, duration: 1000, delay: 200 }} // ✅ Smooth animations
            />
          </SeriesCollectionDirective>
        </ChartComponent>
      )}

      {/* Display additional stock information */}
      <div className="chart-info">
        <h3 className="chart-title">{ticker}</h3>
        <div className="chart-stats">
          <p><strong>Volatility:</strong> {additionalData?.volatility || "N/A"}</p>
          <p><strong>Change:</strong> {additionalData?.change || "N/A"}</p>
          <p><strong>RSI:</strong> {additionalData?.rsi || "N/A"}</p>
        </div>
        <div className="price-info">
          <p><strong>Entry Point:</strong> ${entryPoint?.toFixed(2) || "N/A"}</p>
          <p><strong>Exit Point:</strong> ${exitPoint?.toFixed(2) || "N/A"}</p>
        </div>
      </div>
    </div>
  );
};

export default CandlestickChart;
