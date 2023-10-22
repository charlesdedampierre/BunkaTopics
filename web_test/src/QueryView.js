import React, { useState } from 'react';
import { Typography, Container, Box, Button, TextField, CircularProgress } from '@mui/material';

const QueryView = ({ onQuerySubmit }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedColumnName, setSelectedColumnName] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
    };

    const handleColumnNameChange = (e) => {
        setSelectedColumnName(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!selectedFile || !selectedColumnName) {
            alert('Please select a CSV file and enter a column name.');
            return;
        }

        setIsLoading(true);

        // Call the onQuerySubmit callback with the selected file and column name
        await onQuerySubmit(selectedFile, selectedColumnName);

        setIsLoading(false);
    };

    return (
        <Container>
            <Typography variant="h4" gutterBottom>
                CSV Query View
            </Typography>
            <form onSubmit={handleSubmit}>
                <Box marginBottom={2}>
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        required
                    />
                </Box>
                <Box marginBottom={2}>
                    <TextField
                        label="Enter column name"
                        variant="outlined"
                        fullWidth
                        value={selectedColumnName}
                        onChange={handleColumnNameChange}
                        required
                    />
                </Box>
                <Box>
                    <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <CircularProgress size={24} />
                        ) : (
                            'Analyze'
                        )}
                    </Button>
                </Box>
            </form>
        </Container>
    );
};

export default QueryView;
