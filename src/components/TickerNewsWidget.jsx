import React, { useEffect, useState } from "react";

const TickerNewsWidget = ({ tickers }) => {
  const [news, setNews] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNewsForTickers = async () => {
      setLoading(true);
      const allNews = {};
  
      try {
        for (const ticker of tickers) {
          const response = await fetch(`http://localhost:5000/api/ticker-news?ticker=${ticker}`);
          const data = await response.json();
          console.log(`Full API response for ${ticker}:`, data);
  
          // Access the array under the dynamic key
          const tickerNews = data[ticker]; // Adjusted to access the dynamic key
          if (tickerNews && Array.isArray(tickerNews) && tickerNews.length > 0) {
            allNews[ticker] = tickerNews.map((article, index) => {
              console.log(`Processing article ${index} for ${ticker}:`, article);
              return {
                id: article.id,
                title: article.title,
                article_url: article.article_url,
                author: article.author,
                description: article.description || "No description available.",
                image_url: article.image_url,
                published_utc: article.published_utc,
                sentiment: article.insights?.[0]?.sentiment || "neutral",
                sentiment_reasoning: article.insights?.[0]?.sentiment_reasoning || "No sentiment reasoning provided.",
                publisher: article.publisher?.name || "Unknown Publisher",
              };
            });
          } else {
            console.warn(`No results found for ${ticker} or response structure is invalid.`);
            allNews[ticker] = [];
          }
        }
  
        console.log("Final compiled news object:", allNews);
        setNews(allNews);
      } catch (error) {
        console.error("Error fetching news:", error);
      } finally {
        setLoading(false);
      }
    };
  
    if (tickers.length > 0) {
      fetchNewsForTickers();
    } else {
      setNews({});
      setLoading(false);
    }
  }, [tickers]);
  

  if (loading) {
    return <div className="news-widget">Loading news...</div>;
  }

  if (Object.keys(news).length === 0) {
    return <div className="news-widget">No news available for the selected stocks.</div>;
  }

  return (
    <div className="news-widget">
      <h4>Latest Stock News</h4>
      {Object.entries(news).map(([ticker, articles]) => (
        <div key={ticker} className="news-section">
          <h5>{ticker}</h5>
          {articles.length > 0 ? (
            articles.map((article) => (
              <div key={article.id} className="news-article">
                <h6>
                  <a href={article.article_url} target="_blank" rel="noopener noreferrer">
                    {article.title}
                  </a>
                </h6>
                <p><strong>Author:</strong> {article.author}</p>
                <p><strong>Publisher:</strong> {article.publisher}</p>
                <p><strong>Published:</strong> {new Date(article.published_utc).toLocaleString()}</p>
                {article.description && <p>{article.description}</p>}
                {article.image_url && <img src={article.image_url} alt={article.title} style={{ maxWidth: "100%" }} />}
                <p>
                  <strong>Sentiment:</strong> {article.sentiment} - {article.sentiment_reasoning}
                </p>
              </div>
            ))
          ) : (
            <p>No news available for {ticker}.</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default TickerNewsWidget;