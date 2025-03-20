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

// ✅ Register the Syncfusion license
registerLicense("Ngo9BigBOggjHTQxAR8/V1NMaF5cXmRCf1FpRmJGdld5fUVHYVZUTXxaS00DNHVRdkdmWX5ednVWQ2BfVEJ+WEY=");

const CandlestickChart = ({ ticker, entryPoint, exitPoint }) => {
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  // 🟢 Debugging logs
  console.log(`📌 DEBUG: CandlestickChart received - Ticker: ${ticker}, Entry: ${entryPoint}, Exit: ${exitPoint}`);

  useEffect(() => {
    console.log(`📌 Fetching candlestick data for ${ticker}...`);
    setLoading(true);

    const fetchChartData = async () => {
      try {
        let response = await fetch(`http://localhost:5000/api/candlestick?ticker=${ticker}`);
        let data = await response.json();

        console.log(`✅ API Response for ${ticker}:`, data); 

        if (data.error) {
          setError(data.error);
          setLoading(false);
          return;
        }

        if (data.dates && Array.isArray(data.dates) && data.dates.length > 0) {
          const formattedData = data.dates.map((date, i) => ({
            x: new Date(date),
            open: data.open[i] || 0,
            high: data.high[i] || 0,
            low: data.low[i] || 0,
            close: data.close[i] || 0,
          }));

          setChartData(formattedData);
          setError(""); // ✅ Clear errors
        } else {
          console.warn(`⚠️ No valid historical data available for ${ticker}.`);
          setError("No valid historical data available for this stock.");
        }

      } catch (err) {
        console.error(`❌ Error fetching data for ${ticker}:`, err);
        setError("Failed to load chart data.");
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [ticker]);

  return (
    <div className={`chart-container ${loading ? "loading" : ""}`}>
      {loading ? (
        <p style={{ textAlign: "center" }}>Loading data...</p>
      ) : error ? (
        <p style={{ color: "red", textAlign: "center" }}>{error}</p>
      ) : (
        <>
          <ChartComponent
            id={`chart-${ticker}`}
            primaryXAxis={{ valueType: "DateTime", labelFormat: "MMM dd", intervalType: "Days" }}
            primaryYAxis={{ labelFormat: "${value}" }}
            tooltip={{ enable: true }}
            crosshair={{ enable: true, lineType: "Both" }}
            zoomSettings={{ enableMouseWheelZooming: true, mode: "XY" }}
            height={"400px"}
            width={"100%"}
            legendSettings={{ visible: true }}
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
                animation={{ enable: true, duration: 1000, delay: 200 }}
              />
            </SeriesCollectionDirective>
          </ChartComponent>

          <div className="chart-info">
            <h3 className="chart-title">{ticker}</h3>
            <p>
              <strong>Entry Point:</strong> 
              {entryPoint && !isNaN(entryPoint) ? `$${parseFloat(entryPoint).toFixed(2)}` : "N/A"}
            </p>
            <p>
              <strong>Exit Point:</strong> 
              {exitPoint && !isNaN(exitPoint) ? `$${parseFloat(exitPoint).toFixed(2)}` : "N/A"}
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default CandlestickChart;
