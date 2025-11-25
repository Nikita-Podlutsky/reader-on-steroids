import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Graph, D3Node, D3Link, FilterParams } from '@/types';

interface GraphVisualizationProps {
  data: Graph;
  onNodeClick: (nodeId: string) => void;
  selectedNodeId?: string | null;
  filters?: FilterParams;
  onVisibleCountChange?: (count: number) => void;
}

interface ExtendedD3Node extends D3Node {
  isExpanded?: boolean;
}

export const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  data,
  onNodeClick,
  selectedNodeId,
  filters = {},
  onVisibleCountChange,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const allNodesRef = useRef<ExtendedD3Node[]>([]);
  const visibleNodeIdsRef = useRef<Set<string>>(new Set());
  const generatedLinksRef = useRef<D3Link[]>([]);
  const simulationRef = useRef<d3.Simulation<ExtendedD3Node, D3Link> | null>(null);
  const tooltipRef = useRef<d3.Selection<HTMLDivElement, unknown, HTMLElement, any> | null>(null);
  const currentZoomScaleRef = useRef<number>(0.8); // Текущий масштаб zoom (начальный 0.8)
  const filtersRef = useRef<FilterParams>(filters);
  // Храним ВСЕ узлы, которые когда-либо были раскрыты (для восстановления при расширении фильтров)
  const everRevealedNodesRef = useRef<Set<string>>(new Set());
  // Ref для функции handleNodeClickInternal (чтобы вызывать из других useEffect)
  const handleNodeClickRef = useRef<((node: ExtendedD3Node) => void) | null>(null);

  // Основной useEffect - создаёт граф ОДИН РАЗ при загрузке данных
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !data.nodes.length) return;

    // СБРОС ВСЕХ СОСТОЯНИЙ ПРИ НОВОМ ПОИСКЕ
    visibleNodeIdsRef.current.clear();
    generatedLinksRef.current = [];
    everRevealedNodesRef.current.clear();

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Очистка предыдущего графа
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Создаем группу для зума и панорамирования
    const g = svg.append('g').attr('class', 'main-group');

    // === ЛОГИКА ТИМЛИДА: Сохраняем все узлы, показываем постепенно ===
    
    // 1. Нормализуем координаты чтобы поместились в canvas
    const allX = data.nodes.map(n => n.x);
    const allY = data.nodes.map(n => n.y);
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);
    
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    
    // Масштабируем чтобы занимали 80% canvas с отступами
    const padding = 100;
    const scaleX = (width - 2 * padding) / rangeX;
    const scaleY = (height - 2 * padding) / rangeY;
    const scale = Math.min(scaleX, scaleY);
    
    allNodesRef.current = data.nodes.map((node) => {
      const normalizedX = (node.x - minX) * scale + padding;
      const normalizedY = (node.y - minY) * scale + padding;
      
      return {
        ...node,
        x: normalizedX,
        y: normalizedY,
        fx: normalizedX, // Фиксируем позиции
        fy: normalizedY,
        isExpanded: false,
      };
    });

    // 2. Показываем только root узлы с применением фильтров
    visibleNodeIdsRef.current.clear();
    let rootNodes = allNodesRef.current.filter(n => n.is_root);
    
    // Применяем фильтры только к root-узлам
    if (filters.years && filters.years.length > 0) {
      rootNodes = rootNodes.filter(n => n.year && filters.years!.includes(n.year));
    }
    if (filters.clusters && filters.clusters.length > 0) {
      rootNodes = rootNodes.filter(n => filters.clusters!.includes(n.cluster_id));
    }
    
    const initialNodes = rootNodes.length > 0 ? rootNodes : allNodesRef.current.slice(0, Math.min(5, allNodesRef.current.length));
    initialNodes.forEach(n => {
      visibleNodeIdsRef.current.add(n.id);
      everRevealedNodesRef.current.add(n.id); // Запоминаем начальные узлы
    });

    // 3. Начальные связи - только между видимыми узлами
    generatedLinksRef.current = data.links
      .filter(link => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as D3Node).id;
        const targetId = typeof link.target === 'string' ? link.target : (link.target as D3Node).id;
        return visibleNodeIdsRef.current.has(sourceId) && visibleNodeIdsRef.current.has(targetId);
      })
      .map(link => ({ ...link }));

    // Функция обновления графа
    const updateGraph = () => {
      const visibleNodes = allNodesRef.current.filter(n => visibleNodeIdsRef.current.has(n.id));
      
      // Фильтруем связи: показываем только если оба конца видимы
      const visibleLinks = generatedLinksRef.current.filter(link => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id;
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id;
        return visibleNodeIdsRef.current.has(sourceId) && visibleNodeIdsRef.current.has(targetId);
      });

      // Обновляем симуляцию
      if (simulationRef.current) {
        simulationRef.current.nodes(visibleNodes);
        const linkForce = simulationRef.current.force<d3.ForceLink<ExtendedD3Node, D3Link>>('link');
        if (linkForce) {
          linkForce.links(visibleLinks);
        }
        simulationRef.current.alpha(0.3).restart();
      }

      // Обновляем связи
      const linkSelection = g.selectAll<SVGLineElement, D3Link>('line.link')
        .data(visibleLinks, (d: D3Link) => {
          const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
          const targetId = typeof d.target === 'string' ? d.target : d.target.id;
          return `${sourceId}-${targetId}`;
        });

      linkSelection.exit().remove();

      linkSelection.enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke', 'rgba(200, 200, 200, 0.4)')
        .attr('stroke-width', 2);

      // Обновляем узлы
      const nodeSelection = g.selectAll<SVGGElement, ExtendedD3Node>('g.node-group')
        .data(visibleNodes, (d: ExtendedD3Node) => d.id);

      nodeSelection.exit().remove();

      const nodeEnter = nodeSelection.enter()
        .append('g')
        .attr('class', 'node-group cursor-pointer');
      // Перетаскивание отключено - узлы не двигаются

      // Круги (радиус 8px = диаметр 16px, компенсируется zoom)
      nodeEnter.append('circle')
        .attr('r', 8 / currentZoomScaleRef.current)
        .attr('fill', (d) => d.color)
        .attr('stroke', (d) => d.id === selectedNodeId ? 'white' : d.color)
        .attr('stroke-width', (d) => (d.id === selectedNodeId ? 3 : 1.5) / currentZoomScaleRef.current)
        .on('mouseover', function(_event, d) {
          d3.select(this)
            .attr('stroke', 'white')
            .attr('stroke-width', 3 / currentZoomScaleRef.current);
          showTooltip(_event, d);
        })
        .on('mouseout', function(_event, d) {
          d3.select(this)
            .attr('stroke', d.id === selectedNodeId ? 'white' : d.color)
            .attr('stroke-width', (d.id === selectedNodeId ? 3 : 1.5) / currentZoomScaleRef.current);
          hideTooltip();
        })
        .on('click', (event, d) => {
          event.stopPropagation();
          handleNodeClickInternal(d);
        });

      // Подписи (первые 15 символов названия)
      nodeEnter.append('text')
        .text((d) => {
          const label = d.label || d.topic || d.cluster_name;
          return label.length > 15 ? label.substring(0, 15) + '...' : label;
        })
        .attr('dy', 25 / currentZoomScaleRef.current) // Фиксированное расстояние от узла
        .attr('text-anchor', 'middle')
        .attr('font-size', 10 / currentZoomScaleRef.current) // Фиксированный размер текста
        .attr('class', 'fill-white/80 pointer-events-none')
        .style('opacity', 0.7);
    };

    // === ЛОГИКА ФОНАРИКА: Раскрытие ближайших узлов ===
    const handleNodeClickInternal = (clickedNode: ExtendedD3Node) => {
      // 1. Открываем детали в сайдбаре
      onNodeClick(clickedNode.id);

      // 2. Если узел уже раскрыт - позволяем повторное раскрытие (сбрасываем флаг)
      clickedNode.isExpanded = false;

      // 3. Находим все скрытые узлы
      const hiddenNodes = allNodesRef.current.filter(n => !visibleNodeIdsRef.current.has(n.id));
      
      if (hiddenNodes.length === 0) return;

      // 4. Вычисляем расстояния до скрытых узлов
      hiddenNodes.forEach(node => {
        const dx = (node.x || 0) - (clickedNode.x || 0);
        const dy = (node.y || 0) - (clickedNode.y || 0);
        (node as any)._distance = Math.sqrt(dx * dx + dy * dy);
      });

      // 5. Сортируем по расстоянию и берём 6 ближайших
      hiddenNodes.sort((a: any, b: any) => a._distance - b._distance);
      const nearest6 = hiddenNodes.slice(0, 6);
      
      // 6. Применяем фильтры к 6 ближайшим
      const currentFilters = filtersRef.current;
      let toReveal = nearest6;
      
      if (currentFilters.years && currentFilters.years.length > 0) {
        toReveal = toReveal.filter(n => n.year && currentFilters.years!.includes(n.year));
      }
      if (currentFilters.clusters && currentFilters.clusters.length > 0) {
        toReveal = toReveal.filter(n => currentFilters.clusters!.includes(n.cluster_id));
      }
      
      if (toReveal.length === 0) return;

      // 7. Показываем узлы, прошедшие фильтр
      toReveal.forEach(node => {
        visibleNodeIdsRef.current.add(node.id);
        everRevealedNodesRef.current.add(node.id); // Запоминаем все раскрытые узлы
        
        // Размещаем вокруг кликнутого узла
        const angle = Math.random() * 2 * Math.PI;
        const distance = 100 + Math.random() * 50;
        node.x = (clickedNode.x || 0) + Math.cos(angle) * distance;
        node.y = (clickedNode.y || 0) + Math.sin(angle) * distance;
        
        // Создаём связь от источника
        generatedLinksRef.current.push({
          source: clickedNode.id,
          target: node.id,
        });
      });

      // 8. Обновляем граф
      
      // Уведомляем родителя о количестве видимых узлов
      onVisibleCountChange?.(visibleNodeIdsRef.current.size);
      
      updateGraph();
    };
    
    // Сохраняем функцию в ref для использования из других useEffect
    handleNodeClickRef.current = handleNodeClickInternal;

    // Создаем силовую симуляцию (минимальные силы, координаты фиксированы)
    const simulation = d3.forceSimulation<ExtendedD3Node>(allNodesRef.current.filter(n => visibleNodeIdsRef.current.has(n.id)))
      .force('link', d3.forceLink<ExtendedD3Node, D3Link>(generatedLinksRef.current)
        .id(d => d.id)
        .distance(50)
        .strength(0.1))  // Слабая сила связей
      .force('charge', d3.forceManyBody().strength(-50))  // Слабое отталкивание
      .force('collision', d3.forceCollide().radius(15))
      .alphaDecay(0.5);  // Быстро останавливаем симуляцию

    simulationRef.current = simulation;

    // Позиции уже зафиксированы через fx/fy

    // Начальная отрисовка
    updateGraph();
    
    // Перезапускаем симуляцию чтобы применить tick к новым узлам
    simulation.alpha(0.3).restart();

    // Tooltip - создаем или используем существующий
    if (!tooltipRef.current) {
      tooltipRef.current = d3
        .select('body')
        .append('div')
        .attr('class', 'tooltip absolute bg-black/90 text-white p-3 rounded-lg text-sm pointer-events-none opacity-0 transition-opacity z-50')
        .style('max-width', '300px');
    }
    const tooltip = tooltipRef.current;

    function showTooltip(event: MouseEvent, d: D3Node) {
      tooltip
        .html(
          `
          <strong>${d.label}</strong><br/>
          <span class="text-gray-400">Кластер: ${d.cluster_name}</span><br/>
          ${d.year ? `<span class="text-gray-400">Год: ${d.year}</span>` : ''}
        `
        )
        .style('left', event.pageX + 15 + 'px')
        .style('top', event.pageY - 28 + 'px')
        .style('opacity', '1');
    }

    function hideTooltip() {
      tooltip.style('opacity', '0');
    }

    // Zoom и панорамирование
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        
        // Сохраняем текущий масштаб
        const currentScale = event.transform.k;
        currentZoomScaleRef.current = currentScale;
        
        // Компенсируем zoom для размера узлов и обводки - они остаются фиксированного размера
        const fixedRadius = 8; // Фиксированный радиус в пикселях (диаметр 16px)
        const fixedStrokeWidth = 1.5;
        const fixedStrokeWidthHover = 3;
        
        g.selectAll<SVGCircleElement, ExtendedD3Node>('circle')
          .attr('r', fixedRadius / currentScale)
          .attr('stroke-width', function() {
            // Проверяем если это выбранный узел или при hover (stroke='white')
            const currentStroke = d3.select(this).attr('stroke');
            const isHighlighted = currentStroke === 'white';
            return (isHighlighted ? fixedStrokeWidthHover : fixedStrokeWidth) / currentScale;
          });
        
        // Компенсируем zoom для размера текста и расстояния от узла
        g.selectAll<SVGTextElement, ExtendedD3Node>('text')
          .attr('font-size', 10 / currentScale)
          .attr('dy', 25 / currentScale); // Фиксированное расстояние 25px от центра узла
      });

    svg.call(zoom);

    // Начальный зум по центру графа
    const initialTransform = d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(0.8)
      .translate(-width / 2, -height / 2);
    
    svg.call(zoom.transform as any, initialTransform);

    // Обновление позиций при симуляции
    simulation.on('tick', () => {
      g.selectAll<SVGLineElement, D3Link>('line.link')
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      g.selectAll<SVGGElement, ExtendedD3Node>('g.node-group')
        .attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Функции для перетаскивания удалены - узлы зафиксированы

    // Cleanup
    return () => {
      simulation.stop();
      if (tooltipRef.current) {
        tooltipRef.current.remove();
        tooltipRef.current = null;
      }
    };
  }, [data, onNodeClick]); // Убрали filters из зависимостей!

  // Отдельный эффект для применения фильтров БЕЗ пересоздания графа
  useEffect(() => {
    if (!allNodesRef.current.length) return;
    
    // Обновляем ref с актуальными фильтрами
    filtersRef.current = filters;
    
    // ЛОГИКА ФИЛЬТРАЦИИ: проверяем ВСЕ когда-либо раскрытые узлы
    const newVisibleIds = new Set<string>();
    
    // Проходим по ВСЕМ раскрытым узлам (не только текущим видимым) и проверяем фильтры
    everRevealedNodesRef.current.forEach(nodeId => {
      const node = allNodesRef.current.find(n => n.id === nodeId);
      if (!node) return;
      
      // Проверяем соответствие фильтрам
      let matches = true;
      
      if (filters.years && filters.years.length > 0) {
        matches = matches && node.year !== undefined && filters.years.includes(node.year);
      }
      
      if (filters.clusters && filters.clusters.length > 0) {
        matches = matches && filters.clusters.includes(node.cluster_id);
      }
      
      // Если узел проходит фильтры - показываем его
      if (matches) {
        newVisibleIds.add(nodeId);
      }
    });
    
    // Обновляем список видимых узлов
    visibleNodeIdsRef.current = newVisibleIds;
    
    // Уведомляем родителя о количестве видимых узлов
    onVisibleCountChange?.(visibleNodeIdsRef.current.size);
    
    // Фильтруем ВСЕ связи - оставляем только те, где оба узла видимы
    generatedLinksRef.current = generatedLinksRef.current.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id;
      const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id;
      return visibleNodeIdsRef.current.has(sourceId) && visibleNodeIdsRef.current.has(targetId);
    });
    
    // Обновляем граф с новыми видимыми узлами
    const svg = d3.select(svgRef.current);
    const g = svg.select('g.main-group');
    
    if (g.empty()) return; // Граф ещё не создан
    
    const visibleNodes = allNodesRef.current.filter(n => visibleNodeIdsRef.current.has(n.id));
    
    // Обновляем связи
    const linkSelection = g.selectAll<SVGLineElement, D3Link>('line.link')
      .data(generatedLinksRef.current, (d: D3Link) => {
        const sourceId = typeof d.source === 'string' ? d.source : (d.source as any).id;
        const targetId = typeof d.target === 'string' ? d.target : (d.target as any).id;
        return `${sourceId}-${targetId}`;
      });
    
    linkSelection.exit().remove();
    
    linkSelection.enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke', 'rgba(200, 200, 200, 0.4)')
      .attr('stroke-width', 2);
    
    // Обновляем узлы
    const nodeSelection = g.selectAll<SVGGElement, ExtendedD3Node>('g.node-group')
      .data(visibleNodes, (d: ExtendedD3Node) => d.id);
    
    nodeSelection.exit().remove();
    
    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'node-group cursor-pointer');
    // Перетаскивание отключено
    
    // Круги для новых узлов (радиус 8px = диаметр 16px)
    nodeEnter.append('circle')
      .attr('r', 8 / currentZoomScaleRef.current)
      .attr('fill', (d) => d.color)
      .attr('stroke', (d) => d.color)
      .attr('stroke-width', 1.5 / currentZoomScaleRef.current)
      .on('mouseover', function(_event, d) {
        d3.select(this)
          .attr('stroke', 'white')
          .attr('stroke-width', 3 / currentZoomScaleRef.current);
        if (tooltipRef.current) {
          tooltipRef.current
            .html(
              `
              <strong>${d.label}</strong><br/>
              <span class="text-gray-400">Кластер: ${d.cluster_name}</span><br/>
              ${d.year ? `<span class="text-gray-400">Год: ${d.year}</span>` : ''}
            `
            )
            .style('left', (_event.pageX + 15) + 'px')
            .style('top', (_event.pageY - 28) + 'px')
            .style('opacity', '1');
        }
      })
      .on('mouseout', function(_event, d) {
        d3.select(this)
          .attr('stroke', d.color)
          .attr('stroke-width', 1.5 / currentZoomScaleRef.current);
        if (tooltipRef.current) {
          tooltipRef.current.style('opacity', '0');
        }
      })
      .on('click', (event, d) => {
        event.stopPropagation();
        // Вызываем логику фонарика через ref
        if (handleNodeClickRef.current) {
          handleNodeClickRef.current(d);
        }
      });
    
    // Подписи для новых узлов (первые 15 символов названия)
    nodeEnter.append('text')
      .text((d) => {
        const label = d.label || d.topic || d.cluster_name;
        return label.length > 15 ? label.substring(0, 15) + '...' : label;
      })
      .attr('dy', 25 / currentZoomScaleRef.current) // Фиксированное расстояние от узла
      .attr('text-anchor', 'middle')
      .attr('font-size', 10 / currentZoomScaleRef.current) // Фиксиро��анный размер текста
      .attr('class', 'fill-white/80 pointer-events-none')
      .style('opacity', 0.7);
    
    // Перезапускаем симуляцию с обновленными узлами и связями
    if (simulationRef.current) {
      simulationRef.current.nodes(visibleNodes);
      const linkForce = simulationRef.current.force<d3.ForceLink<ExtendedD3Node, D3Link>>('link');
      if (linkForce) {
        linkForce.links(generatedLinksRef.current);
      }
      simulationRef.current.alpha(0.3).restart();
    }
  }, [filters, onVisibleCountChange]);

  // Отдельный эффект для обновления подсветки выбранного узла
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const currentScale = currentZoomScaleRef.current;
    
    // Обновляем обводку всех узлов с учетом текущего масштаба
    svg.selectAll<SVGCircleElement, ExtendedD3Node>('circle')
      .attr('stroke', (d) => d.id === selectedNodeId ? 'white' : d.color)
      .attr('stroke-width', (d) => (d.id === selectedNodeId ? 3 : 1.5) / currentScale);
  }, [selectedNodeId]);

  return (
    <div ref={containerRef} className="w-full h-full bg-[#0a0e17]">
      <svg ref={svgRef} className="w-full h-full bg-[#0a0e17]" />
    </div>
  );
};
