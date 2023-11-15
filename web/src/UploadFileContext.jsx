import React, { useState, createContext, useEffect } from 'react';
import axios from 'axios'
import useSWR from 'swr';
import  md5 from 'crypto-js/md5';
import { Alert } from "@mui/material";

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
/*
const saveDataToFile = (fileName, data) => {
  const blob = new Blob([data], { type: "application/json" });

  // Create a link element
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName;

  // Trigger a click event to download the file
  a.click();
};
*/

// Fetcher function for useSWR
const fetcher = (url, data) => axios.post(url, data).then(res => res.data)

// Provider Component
export const TopicsProvider = ({ children, onSelectView }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [errorText, setErrorText] = useState('');

  // Use SWR for data fetching
  const { data, mutate, error } = useSWR('/topics', fetcher, { shouldRetryOnError: false });
  
  // Handle File Upload and POST Request
  const uploadFile = async (file, params) => {
    setIsLoading(true);
    setErrorText('');
    
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
      await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      onSelectView("map");
      // Save topics and docs to files in the public directory
      // saveDataToFile("bunka_topics.json", JSON.stringify(data.topics));
      // saveDataToFile("bunka_docs.json", JSON.stringify(data.docs));
  
      // Trigger a revalidation with the new URL
      mutate(`/topics?md5=${fileHash}`);
    } catch (errorExc) {
      setErrorText(`Error uploading file. Please try later : ${errorExc}`);
      console.error('Error uploading file:', errorExc);
    } finally {
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    if (error !== undefined && error.length) {
      const message = error.response ? error.response.data.message : error.message;
      setErrorText(`Error uploading file.\n${message}`);
      console.error('Error uploading file:', message);
    }
  }, [error]);

  return (
    <TopicsContext.Provider value={{ data, uploadFile, isLoading, error }}>
      <>
        {isLoading && <div className="loader"></div>}
        {errorText &&
          <Alert severity="error" className="errorMessage">{errorText}</Alert>
        }
        {children}
      </>
    </TopicsContext.Provider>
  );
};
