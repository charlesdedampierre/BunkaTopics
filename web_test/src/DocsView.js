import React, { useEffect, useState } from 'react';
import {
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Box,
    Container,
} from '@mui/material';

const DocsView = () => {
    const [docs, setDocs] = useState(null);
    const [topics, setTopics] = useState(null);

    useEffect(() => {
        // Fetch the content of "docs.json" when the component mounts
        fetch('/bunka_docs.json')
            .then((response) => response.json())
            .then((data) => {
                setDocs(data);
            })
            .catch((error) => {
                console.error('Error fetching JSON data:', error);
            });

        // Fetch the topics data when the component mounts
        fetch('/bunka_topics.json')
            .then((response) => response.json())
            .then((data) => {
                setTopics(data);
            })
            .catch((error) => {
                console.error('Error fetching topics data:', error);
            });
    }, []);

    const docsWithTopics = docs && topics
        ? docs.map((doc) => ({
            ...doc,
            topic_name: topics.find((topic) => topic.topic_id === doc.topic_id)?.name || 'Unknown',
        }))
        : [];

    const downloadCSV = () => {
        // Create a CSV content string from the data
        const csvContent = "data:text/csv;charset=utf-8," + [
            ['Doc ID', 'Topic ID', 'Topic Name', 'Content'], // CSV header
            ...docsWithTopics.map((doc) => [doc.doc_id, doc.topic_id, doc.topic_name, doc.content]), // CSV data
        ]
            .map((row) => row.map((cell) => `"${cell}"`).join(',')) // Wrap cells in double quotes
            .join('\n'); // Join rows with newline

        // Create a Blob containing the CSV data
        const blob = new Blob([csvContent], { type: 'text/csv' });

        // Create a download URL for the Blob
        const url = URL.createObjectURL(blob);

        // Create a temporary anchor element to trigger the download
        const a = document.createElement('a');
        a.href = url;
        a.download = 'docs.csv'; // Set the filename for the downloaded file
        a.click();

        // Revoke the URL to free up resources
        URL.revokeObjectURL(url);
    };

    return (
        <Container fixed>
            <div className="docs-view">
                <h2>Documents View</h2>
                {docs ? (
                    <div>
                        <Box
                            sx={{
                                height: '1000px', // Set the height of the table
                                overflow: 'auto', // Add scroll functionality
                            }}
                        >
                            <TableContainer component={Paper}>
                                <Table>
                                    <TableHead
                                        sx={{
                                            backgroundColor: 'lightblue', // Set background color
                                            position: 'sticky', // Make the header sticky
                                            top: 0, // Stick to the top
                                        }}
                                    >
                                        <TableRow>
                                            <TableCell>Doc ID</TableCell>
                                            <TableCell>Topic ID</TableCell>
                                            <TableCell>Topic Name</TableCell>
                                            <TableCell>Content</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {docsWithTopics.map((doc, index) => (
                                            <TableRow
                                                key={index}
                                                sx={{
                                                    borderBottom: '1px solid lightblue', // Add light blue border
                                                }}
                                            >
                                                <TableCell>{doc.doc_id}</TableCell>
                                                <TableCell>{doc.topic_id}</TableCell>
                                                <TableCell>{doc.topic_name}</TableCell>
                                                <TableCell>{doc.content}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Box>
                        <Button variant="contained" color="primary" onClick={downloadCSV}>
                            Download CSV
                        </Button>
                    </div>
                ) : (
                    <p>Loading...</p>
                )}
            </div>
        </Container>
    );
};

export default DocsView;
