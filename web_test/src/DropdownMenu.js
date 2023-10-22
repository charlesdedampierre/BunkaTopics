import React from 'react';
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';

const DropdownMenu = ({ onSelectView }) => {
    const handleSelectView = (event) => {
        onSelectView(event.target.value);
    };

    return (
        <FormControl variant="outlined" className="dropdown-menu" sx={{ minWidth: '200px', marginTop: '85px' }}>
            <InputLabel htmlFor="view-select">Select a View</InputLabel>
            <Select
                label="Select a View"
                value=""
                onChange={handleSelectView}
                inputProps={{
                    name: 'view-select',
                    id: 'view-select',
                }}
            >
                <MenuItem value="map">Map View</MenuItem>
                <MenuItem value="docs">Documents View</MenuItem>
                <MenuItem value="import">Import CSV</MenuItem> {/* Add a new view for importing CSV */}
            </Select>
        </FormControl>
    );
};

export default DropdownMenu;
