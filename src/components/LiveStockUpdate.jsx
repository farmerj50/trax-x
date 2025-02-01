import React, { useEffect, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://localhost:5000"); // Adjust backend URL if needed

const LiveStockUpdates = () => {
    const [stocks, setStocks] = useState({});

    useEffect(() => {
        socket.on("stock_update", (data) => {
            console.log("📡 Live Update Received:", data); // Debugging logs
            setStocks((prevStocks) => ({
                ...prevStocks,
                [data.ticker]: data.price, 
            }));
        });

        return () => {
            socket.off("stock_update");
        };
    }, []);

    return (
        <div>
            <h2>Live Stock Updates</h2>
            <ul>
                {Object.keys(stocks).map((ticker) => (
                    <li key={ticker}>
                        {ticker}: <strong>${stocks[ticker]?.toFixed(2)}</strong>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default LiveStockUpdates;
