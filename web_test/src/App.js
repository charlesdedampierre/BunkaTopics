import React, { useState } from 'react';
import DropdownMenu from './DropdownMenu';
import DocsView from './DocsView';
import CsvUpload from './CsvUpload';
import CSVView from './CSVView';
import Map from './Map'; // Import the MapView component

const App = () => {
  const [selectedView, setSelectedView] = useState('map'); // Default to 'map'

  // Define a state to hold CSV data
  const [csvData, setCSVData] = useState(null);

  // Callback function to handle CSV import
  const handleCSVImport = (data) => {
    // Process the CSV data as needed
    setCSVData(data);

    // Set the selected view back to 'map' or any other desired view
    setSelectedView('map');
  };

  return (
    <div className="App">
      <div className="json-display">
        <div className="top-right">
          <a
            href="https://www.linkedin.com/company/company-profile-link"
            target="_blank"
            rel="noopener noreferrer"
            className="linkedin-icon"
          >
            <img src="/linkedin_logo.png" alt="LinkedIn" />
          </a>
          <img src="/bunka_logo.png" alt="Bunka Logo" className="bunka-logo" />
          <DropdownMenu onSelectView={setSelectedView} />
        </div>
        {selectedView === 'map' ? (
          <Map /> // Render the MapView component with "Map" title
        ) : selectedView === 'docs' ? (
          <DocsView />
        ) : (
          // Render the CSV import view when 'import' is selected
          <CSVView onCSVImport={handleCSVImport} />
        )}
      </div>
    </div>
  );
};

export default App;
