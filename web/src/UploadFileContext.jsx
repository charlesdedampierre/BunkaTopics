import React, { useState, createContext } from 'react';
import useSWR from 'swr';
import  md5 from 'crypto-js/md5';

// Create the Context
export const TopicsContext = createContext();

/**
 * Return a MD5 from the contents of a text file.
 * @param {Promise<FileReader>} file 
 * @returns String
 */
async function hashPartialFile(file) {
  const reader = new FileReader();
  reader.readAsText(file);

  return new Promise((resolve, reject) => {
    reader.onload = async (event) => {
      const text = event.target.result;
      const lines = text.split('\n');

      let contentToHash;
      if (lines.length < 5) {
        // If less than 5 lines, use the entire content
        contentToHash = text;
      } else {
        // Otherwise, use the first two and the last two lines
        contentToHash = [...lines.slice(0, 2), ...lines.slice(-2)].join('\n');
      }

      // Compute MD5 hash
      const hash = md5(contentToHash).toString();
      resolve(hash);
    };

    reader.onerror = (error) => {
      reject(error);
    };
  });
}

/**
 * Trigger a download of a file
 * @param {String} fileName 
 * @param {Object} data 
 */
const saveDataToFile = (fileName, data) => {
  const blob = new Blob([data], { type: "application/json" });

  // Create a link element
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName;

  // Trigger a click event to download the file
  a.click();
};

// Fetcher function for useSWR
const fetcher = url => fetch(url).then(res => res.json());

// Provider Component
export const TopicsProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Use SWR for data fetching
  const { data, mutate } = useSWR('/topics', fetcher);

  // Handle File Upload and POST Request
  const uploadFile = async (file, params) => {
    setIsLoading(true);
    setError('');
  
    try {
      // Generate SHA-256 hash of the file
      const fileHash = await hashPartialFile(file);
  
      const formData = new FormData();
      formData.append('file', file);
      // Append additional parameters to formData
      formData.append('n_cluster', params.n_cluster);
      formData.append('openapi_key', params.openapi_key);
      formData.append('selected_column', params.selected_column);

      const apiUrl = `${process.env.REACT_APP_API_ENDPOINT}/topics?md5=${fileHash}`;
      // Perform the POST request
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });
  
      if (response.ok) {
        const data = await response.json();
  
        // Save topics and docs to files in the public directory
        saveDataToFile("bunka_topics.json", JSON.stringify(data.topics));
        saveDataToFile("bunka_docs.json", JSON.stringify(data.docs));
      }
      // Trigger a revalidation with the new URL
      mutate(`/topics?md5=${fileHash}`);
    } catch (error) {
      setError('Error uploading file. Please try again.');
      console.error('Error uploading file:', error);
    } finally {
      setIsLoading(false);
    }
  };
  

  return (
    <TopicsContext.Provider value={{ data, uploadFile, isLoading, error }}>
      {isLoading && <div className="loader"></div>}
      {error && <div className="errorMessage">{error}</div>}
      {children}
    </TopicsContext.Provider>
  );
};
