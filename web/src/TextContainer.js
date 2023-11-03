import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import DescriptionIcon from '@mui/icons-material/Description'; // Import the document icon

const TextContainer = (props) => {
    const [selectedDocument, setSelectedDocument] = useState(null);

    const handleDocumentClick = (docIndex) => {
        if (selectedDocument === docIndex) {
            setSelectedDocument(null);
        } else {
            setSelectedDocument(docIndex);
        }
    };

    return (
        <Box className="topic-box">
            <Box
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    background: 'rgb(94, 163, 252)',
                    padding: '8px',
                    color: 'white',
                    textAlign: 'center',
                    borderRadius: '20px', // Make it rounder
                    margin: '0 auto', // Center horizontally
                    width: '80%', // Adjust the width as needed
                }}
            >
                <Typography variant="h4" style={{ marginBottom: '8px' }}>
                    {props.topicName}
                </Typography>

            </Box>
            <Typography variant="h5" style={{ marginBottom: '20px', marginTop: '20px', textAlign: 'center' }}>
                {props.sizeFraction}% of the Territory
            </Typography>
            <Paper elevation={3} style={{ maxHeight: '900px', overflowY: 'auto' }}>
                <List>
                    {props.content.map((doc, index) => (
                        <ListItem
                            button
                            key={index}
                            onClick={() => handleDocumentClick(index)}
                            selected={selectedDocument === index}
                        >
                            <ListItemIcon>
                                <DescriptionIcon /> {/* Display a document icon */}
                            </ListItemIcon>
                            <ListItemText primary={<span style={{ fontSize: '16px' }}>{doc}</span>} />
                        </ListItem>
                    ))}
                </List>
            </Paper>
        </Box >
    );
};

export default TextContainer;