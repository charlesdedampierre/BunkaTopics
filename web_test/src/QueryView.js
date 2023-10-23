import React, { useState } from 'react';
import Papa from 'papaparse';
import FileSaver from 'file-saver'; // Import the FileSaver library
import {
    Typography,
    Container,
    Box,
    Button,
    CircularProgress,
    Table,
    TableContainer,
    TableHead,
    TableBody,
    TableRow,
    TableCell,
    Paper,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
} from '@mui/material';

const QueryView = () => {
    const [fileData, setFileData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedColumn, setSelectedColumn] = useState('');
    const [selectedColumnData, setSelectedColumnData] = useState([]);
    const [topics, setTopics] = useState([]); // State to store topics
    const [docs, setDocs] = useState([]); // State to store docs

    const handleFileChange = async (e) => {
        const file = e.target.files[0];

        if (!file) return;

        setIsLoading(true);

        try {
            const parsedData = await parseCSVFile(file);
            setFileData(parsedData);
            setSelectedColumn(''); // Clear the selected column when a new file is uploaded
        } catch (error) {
            console.error('Error parsing CSV:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const parseCSVFile = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (e) => {
                const csvData = e.target.result;
                Papa.parse(csvData, {
                    complete: (result) => {
                        resolve(result.data);
                    },
                    error: (error) => {
                        reject(error.message);
                    },
                });
            };

            reader.readAsText(file);
        });
    };

    const handleColumnSelect = (e) => {
        const columnName = e.target.value;
        setSelectedColumn(columnName);

        // Extract the content of the selected column
        const columnIndex = fileData[0].indexOf(columnName);
        const columnData = fileData.slice(1).map((row) => row[columnIndex]);

        setSelectedColumnData(columnData);
    };

    const handleProcessTopics = async () => {
        if (selectedColumnData.length === 0) return;

        const apiUrl = 'http://localhost:8000/topics/';

        const params = {
            n_cluster: 10, // You can set the desired number of clusters here
        };

        // Transform selectedColumnData into a list of strings
        const full_docs = selectedColumnData.map((item) => String(item));

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ params, full_docs }),
        });

        if (response.ok) {
            const data = await response.json();
            console.log('API Response:', data);

            // Set the topics and docs in the state
            setTopics(data.topics);
            setDocs(data.docs);

            // Save topics and docs to files in the public directory
            saveDataToFile('bunka_topics.json', JSON.stringify(data.topics));
            saveDataToFile('bunka_docs.json', JSON.stringify(data.docs));
        } else {
            console.error('API Request Failed');
        }
    };

    // Function to save data to a file
    const saveDataToFile = (fileName, data) => {
        const jsonData = JSON.stringify(data);
        const blob = new Blob([jsonData], { type: 'application/json' });

        // Create a link element
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = fileName;

        // Trigger a click event to download the file
        a.click();
    };

    return (
        <Container>
            <Typography variant="h4" gutterBottom>
                CSV File Viewer
            </Typography>
            <Box marginBottom={2}>
                <input type="file" accept=".csv" onChange={handleFileChange} required />
            </Box>
            <Box marginBottom={2}>
                <FormControl variant="outlined" fullWidth>
                    <InputLabel>Select a Column</InputLabel>
                    <Select value={selectedColumn} onChange={handleColumnSelect}>
                        {fileData[0] &&
                            fileData[0].map((header, index) => (
                                <MenuItem key={index} value={header}>
                                    {header}
                                </MenuItem>
                            ))}
                    </Select>
                </FormControl>
            </Box>
            {isLoading ? (
                <CircularProgress />
            ) : (
                selectedColumnData.length > 0 && (
                    <TableContainer
                        component={Paper}
                        style={{ maxHeight: '400px', overflowY: 'auto' }}
                    >
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>{selectedColumn}</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {selectedColumnData.map((cell, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{cell}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )
            )}
            <Box marginTop={2}>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleProcessTopics}
                    disabled={selectedColumnData.length === 0}
                >
                    Process Topics
                </Button>
            </Box>
        </Container>
    );
};

export default QueryView;
