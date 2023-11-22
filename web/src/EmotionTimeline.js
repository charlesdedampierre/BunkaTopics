import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';

const YourComponent = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        const fetchData = () => {
            Papa.parse(process.env.PUBLIC_URL + '/df.csv', {
                download: true,
                header: true,
                complete: function (result) {
                    const records = result.data;
                    setData(records);
                },
                error: function (error) {
                    console.error('Error parsing CSV:', error.message);
                },
            });
        };

        fetchData();
    }, []);

    const handleIntersectionClick = (point) => {
        const filteredData = data.filter((d) => d.date === point.date && d.user_input === point.name);
        setSelectedData(filteredData);
    };

    return (
        <div style={{ display: 'flex' }}>
            <div style={{ flex: 1 }}>
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {Array.from(new Set(data.map((d) => d.user_input))).map((emotion) => (
                            <Line
                                key={emotion}
                                type="monotone"
                                dataKey={emotion}
                                stroke={d3.schemeCategory10[data.map((d) => d.user_input).indexOf(emotion)]}
                                activeDot={{ r: 8 }}
                                onClick={(point) => handleIntersectionClick(point)}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div style={{ flex: 1, marginLeft: '20px' }}>
                <Paper elevation={3} style={{ padding: '20px' }}>
                    <Typography variant="h6">Selected Texts</Typography>
                    <List>
                        {selectedData.map((item, index) => (
                            <ListItem key={index}>
                                <ListItemText primary={item.content_clean} />
                            </ListItem>
                        ))}
                    </List>
                </Paper>
            </div>
        </div>
    );
};

export default EmotionTimeline;
