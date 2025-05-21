import React, { useState, useEffect } from 'react';
import './App.css';
import { FaHeartbeat, FaMedkit, FaUtensils, FaRunning, FaExclamationTriangle } from 'react-icons/fa';

function App() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [, setDiseases] = useState([]);
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    // Fetch the list of diseases when component mounts
    fetch('http://localhost:3000/api/diseases')
      .then(response => response.json())
      .then(data => setDiseases(data.diseases))
      .catch(err => console.error('Failed to fetch diseases:', err));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input }),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setInput(example);
  };

  return (
    <div className="app">
      <header className="header">
        <FaHeartbeat size={30} />
        <h1>Healthcare Recommendation System</h1>
      </header>

      <div className="container">
        <section className="input-section">
          <h2>Describe Your Symptoms</h2>
          <p>
            Enter your symptoms or disease description below for personalized recommendations.
            <button 
              className="help-button"
              onClick={() => setShowHelp(!showHelp)}
            >
              ?
            </button>
          </p>
          
          {showHelp && (
            <div className="help-box">
              <h3>How to Describe Your Symptoms</h3>
              <p>For best results, please:</p>
              <ul>
                <li>Be specific about symptoms (pain, fever, cough, etc.)</li>
                <li>Include duration (how long you've had symptoms)</li>
                <li>Mention any triggers or patterns you've noticed</li>
                <li>Include severity (mild, moderate, severe)</li>
              </ul>
              <p>Example: "I have a persistent dry cough for the past week, mild fever at night, and fatigue during the day."</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="input-form">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Example: I've been experiencing severe headache, fever, and joint pain for the past 3 days..."
              rows={5}
            />
            <button type="submit" disabled={loading || !input.trim()}>
              {loading ? 'Processing...' : 'Get Recommendations'}
            </button>
          </form>

          <div className="examples">
            <h3>Examples:</h3>
            <div className="example-buttons">
              <button onClick={() => handleExampleClick("I have a persistent cough, high fever, and difficulty breathing for the past week.")}>
                Respiratory Symptoms
              </button>
              <button onClick={() => handleExampleClick("I'm experiencing severe headache, sensitivity to light, and neck stiffness.")}>
                Neurological Symptoms
              </button>
              <button onClick={() => handleExampleClick("I have a rash with itchy red spots spreading across my torso and arms.")}>
                Skin Symptoms
              </button>
            </div>
          </div>
        </section>

        {error && (
          <div className="error-message">
            <FaExclamationTriangle /> {error}
          </div>
        )}

        {result && (
          <section className="result-section">
            <div className="disease-card">
              <h2>
                <FaHeartbeat /> Predicted Condition
              </h2>
              <h3>{result.disease}</h3>
              <p className="description">{result.description}</p>
              <div className="confidence">
                <div className="confidence-bar">
                  <div 
                    className="confidence-level" 
                    style={{ width: `${Math.min(100, result.confidence * 100)}%` }}
                  ></div>
                </div>
                <span>Confidence: {Math.round(result.confidence * 100)}%</span>
              </div>
            </div>

            <div className="recommendations-grid">
              <div className="rec-card">
                <h3><FaMedkit /> Medications</h3>
                {result.recommendations.Medicines && result.recommendations.Medicines.length > 0 ? (
                  <ul>
                    {result.recommendations.Medicines.map((med, idx) => (
                      <li key={idx}>{med}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-data">No medication data available</p>
                )}
              </div>

              <div className="rec-card">
                <h3><FaUtensils /> Diet Recommendations</h3>
                {result.recommendations.Diets && result.recommendations.Diets.length > 0 ? (
                  <ul>
                    {result.recommendations.Diets.map((diet, idx) => (
                      <li key={idx}>{diet}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-data">No diet recommendations available</p>
                )}
              </div>

              <div className="rec-card">
                <h3><FaExclamationTriangle /> Precautions</h3>
                {result.recommendations.Precautions && result.recommendations.Precautions.length > 0 ? (
                  <ul>
                    {result.recommendations.Precautions.map((prec, idx) => (
                      <li key={idx}>{prec}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-data">No precautions available</p>
                )}
              </div>

              <div className="rec-card">
                <h3><FaRunning /> Exercise Recommendations</h3>
                {result.recommendations.Workouts && result.recommendations.Workouts.length > 0 ? (
                  <ul>
                    {result.recommendations.Workouts.map((workout, idx) => (
                      <li key={idx}>{workout}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-data">No workout recommendations available</p>
                )}
              </div>
            </div>

            <div className="disclaimer">
              <p><strong>Disclaimer:</strong> This is not a substitute for professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

export default App;