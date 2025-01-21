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
} from "@syncfusion/ej2-react-charts";
import { registerLicense } from "@syncfusion/ej2-base";

// Register the Syncfusion license
registerLicense("Ngo9BigBOggjHTQxAR8/V1NMaF5cXmRCf1FpRmJGdld5fUVHYVZUTXxaS00DNHVRdkdmWX5ednVWQ2BfVEJ+WEY=");

const CandlestickChart = ({ ticker, entryPoint, exitPoint, additionalData }) => {
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    // Fetch chart data for the specific ticker
    const fetchChartData = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/candlestick?ticker=${ticker}`);
        const data = await response.json();

        if (data.dates) {
          // Format the data for Syncfusion
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
          setError("No data available for this stock.");
        }
      } catch (err) {
        console.error(`Error fetching data for ${ticker}:`, err);
        setError("Failed to load chart data.");
      }
    };

    fetchChartData();
  }, [ticker]);

  return (
    <div className="chart-container">
      {/* Render error if data fetch fails */}
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
          height="400px"
          width="100%"
        >
          <Inject services={[CandleSeries, DateTime, Tooltip, Zoom, Crosshair]} />
          <SeriesCollectionDirective>
            <SeriesDirective
              dataSource={chartData}
              xName="x"
              open="open"
              high="high"
              low="low"
              close="close"
              type="Candle"
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
          <p><strong>Entry Point:</strong> ${entryPoint.toFixed(2)}</p>
          <p><strong>Exit Point:</strong> ${exitPoint.toFixed(2)}</p>
        </div>
      </div>
    </div>
  );
};

export default CandlestickChart;