import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import {
    Paper,
    Typography,
    List,
    ListItem,
} from '@mui/material';

const TreemapView = () => {
    const svgRef = useRef(null);
    const [topics, setTopics] = useState([]);
    const [selectedTopic, setSelectedTopic] = useState({ name: '', content: [] });

    useEffect(() => {
        // Fetch the topics data when the component mounts
        fetch('/bunka_topics.json')
            .then((response) => response.json())
            .then((data) => {
                setTopics(data);
                createTreemap(data);
            })
            .catch((error) => {
                console.error('Error fetching topics data:', error);
            });
    }, []);

    const createTreemap = (data) => {
        const width = window.innerWidth * 0.6; // Adjust the width for the treemap
        const height = 800; // Adjust the height as needed

        const svg = d3.select(svgRef.current)
            .attr('width', width)
            .attr('height', height);

        const root = d3.hierarchy({ children: data })
            .sum((d) => d.size);

        const treemapLayout = d3.treemap()
            .size([width, height])
            .padding(1)
            .round(true);

        treemapLayout(root);

        const cell = svg.selectAll('g')
            .data(root.leaves())
            .enter()
            .append('g')
            .attr('transform', (d) => `translate(${d.x0},${d.y0})`)
            .on('click', (event, d) => {
                const topicName = d.data.name;
                const topicContent = d.data.top_doc_content || [];

                setSelectedTopic({ name: topicName, content: topicContent });
            });

        cell.append('rect')
            .attr('width', (d) => d.x1 - d.x0)
            .attr('height', (d) => d.y1 - d.y0)
            .style('fill', 'lightblue')
            .style('stroke', 'blue');

        cell.append('text')
            .selectAll('tspan')
            .data((d) => {
                const text = d.data.name.split(/(?=[A-Z][^A-Z])/g); // Split topic name on capital letters
                return text;
            })
            .enter()
            .append('tspan')
            .attr('x', 3)
            .attr('y', (d, i) => 13 + i * 10)
            .text((d) => d);

        svg.selectAll('text')
            .attr('font-size', 13)
            .attr('fill', 'black');
    };

    return (
        <div>
            <h2>Treemap View</h2>
            <div style={{ display: 'flex' }}>
                <svg ref={svgRef} style={{ marginRight: '20px' }}></svg>
                <div style={{ width: window.innerWidth * 0.25, maxHeight: '800px', overflowY: 'auto' }}>
                    <Paper>
                        <Typography variant="h4" style={{ position: 'sticky', top: 0, backgroundColor: 'white', color: 'blue' }}>
                            {selectedTopic.name}
                        </Typography>
                        {selectedTopic.content.map((doc, index) => (
                            <List key={index}>
                                <ListItem>
                                    <Typography variant="h5">{doc}</Typography>
                                </ListItem>
                            </List>
                        ))}
                        {selectedTopic.content.length === 0 && (
                            <Typography variant="h4">Click on a Square.</Typography>
                        )}
                    </Paper>
                </div>
            </div>
        </div>
    );
};

export default TreemapView;