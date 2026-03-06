// Цветовая схема для кластеров
const clusterColorPalette = [
  '#3498db', // синий
  '#9b59b6', // фиолетовый
  '#e74c3c', // красный
  '#2ecc71', // зеленый
  '#f39c12', // оранжевый
  '#1abc9c', // бирюзовый
  '#34495e', // темно-серый
  '#e67e22', // морковный
  '#16a085', // зеленое море
  '#c0392b', // гранат
];

export const getClusterColor = (cluster: number): string => {
  return clusterColorPalette[cluster % clusterColorPalette.length] || '#7f8c8d';
};

// Генерация мок-данных для тестирования
export const generateMockData = () => {
  const topics = [
    'Diffusion Models',
    'Graph Neural Networks',
    'Reinforcement Learning',
    'NLP',
    'Computer Vision',
    'Optimization',
    'Cryptography',
    'Robotics',
  ];

  const nodes = [];
  const links = [];

  // Создаем узлы
  for (let i = 0; i < 50; i++) {
    const cluster = Math.floor(Math.random() * 5);
    nodes.push({
      id: `node-${i}`,
      title: `Paper ${i}: ${topics[Math.floor(Math.random() * topics.length)]}`,
      year: 2018 + Math.floor(Math.random() * 7),
      cluster: cluster,
      color: getClusterColor(cluster),
      topic: topics[Math.floor(Math.random() * topics.length)],
      abstract: `This is an abstract for paper ${i} about ${topics[Math.floor(Math.random() * topics.length)]}.`,
      is_root: i < 5, // Первые 5 узлов - корневые
    });
  }

  // Создаем связи между узлами
  for (let i = 0; i < 80; i++) {
    const source = Math.floor(Math.random() * nodes.length);
    let target;
    do {
      target = Math.floor(Math.random() * nodes.length);
    } while (source === target);

    links.push({
      source: nodes[source].id,
      target: nodes[target].id,
    });
  }

  return { nodes, links };
};
