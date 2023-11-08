import React, { useState } from 'react';
import DropdownMenu from './DropdownMenu';
import DocsView from './DocsView';
import Map from './Map';
import TreemapView from './TreemapView'; // Import the TreemapView component
import QueryView from './QueryView'; // Import the QueryView component
import Bourdieu from './Bourdieu'; // Import the QueryView component

const App = () => {
  const [selectedView, setSelectedView] = useState('map'); // Default to 'map'
  const [csvData, setCSVData] = useState(null);

  const handleCSVImport = (data) => {
    // Process the CSV data as needed
    setCSVData(data);

    // Set the selected view back to 'map' or any other desired view
    setSelectedView('map');
  };

  const handleQuerySubmit = async (csvFile, columnName) => {
    // Perform API request with the CSV file and column name
    // Replace the following code with your actual API request
    const formData = new FormData();
    formData.append('csvFile', csvFile);
    formData.append('columnName', columnName);

    try {
      const response = await fetch('https://your-api-endpoint.com/topics', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      // Handle the API response (topics and docs) as needed
      console.log('API Response:', data);

      // Update the selected view to display the results
      setSelectedView('results');
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <div className="main-display">
        <div className="top-right" id="top-banner">
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
          <Map />
        ) : selectedView === 'docs' ? (
          <DocsView />
        ) : selectedView === 'treemap' ? (
          <TreemapView />
        ) : selectedView === 'query' ? (
          <QueryView />
        ) : selectedView === 'bourdieu' ? (
          <Bourdieu />
        ) : (
          <QueryView onQuerySubmit={handleQuerySubmit} />
        )}
      </div>
    </div>
  );
};

export default App;
