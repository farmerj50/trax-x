import React, { useState } from "react";

const AddTicker = () => {
    const [ticker, setTicker] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!ticker) return;

        try {
            const response = await fetch("http://localhost:5000/api/add_ticker", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ ticker }),
            });

            const data = await response.json();
            alert(data.message); // Show confirmation
        } catch (error) {
            console.error("Error adding ticker:", error);
        }

        setTicker(""); // Reset input field
    };

    return (
        <div>
            <h3>Add Stock to Watchlist</h3>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    placeholder="Enter Stock Symbol"
                    required
                />
                <button type="submit">Add</button>
            </form>
        </div>
    );
};

export default AddTicker;
