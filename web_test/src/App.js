import React from 'react';
import Map from './Map';
import CsvUpload from './CsvUpload'; // Import the CsvUpload component

const App = () => {
  return (
    <div className="App">
      <div className="json-display">
        <div className="top-right">
          <a href="https://www.linkedin.com/company/company-profile-link" target="_blank" rel="noopener noreferrer" className="linkedin-icon">
            <img src="/linkedin_logo.png" alt="LinkedIn" />
          </a>
          <img src="/bunka_logo.png" alt="Bunka Logo" className="bunka-logo" />
        </div>
        <Map />
      </div>
    </div>
  );
}

export default App;