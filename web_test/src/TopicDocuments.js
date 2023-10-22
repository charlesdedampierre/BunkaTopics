import React from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogContentText,
    DialogActions,
    Button,
    Paper,
    List,
    ListItem,
    ListItemText,
    Divider,
} from '@mui/material';

const TopicDocuments = ({ open, onClose, documents }) => {
    return (
        <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
            <DialogTitle>Documents in This Topic</DialogTitle>
            <DialogContent>
                <DialogContentText>
                    Below are the documents contained in this topic:
                </DialogContentText>
                <Paper elevation={3} style={{ maxHeight: '500px', overflow: 'auto' }}>
                    <List>
                        {documents.map((document, index) => (
                            <React.Fragment key={document.doc_id}>
                                <ListItem>
                                    <ListItemText
                                        primary={`Document #${document.doc_id}`}
                                        secondary={document.content}
                                    />
                                </ListItem>
                                {index < documents.length - 1 && <Divider />}
                            </React.Fragment>
                        ))}
                    </List>
                </Paper>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} color="primary">
                    Close
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default TopicDocuments;
