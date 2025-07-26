"""
Módulo de detección de objetos metálicos para el Visualizador de Campos Electromagnéticos
"""

import cv2
import numpy as np


class ObjectDetector:
    def __init__(self):
        # Parámetros para la detección de objetos metálicos
        self.hsv_lower = np.array([0, 0, 90])  # Valores HSV mínimos para objetos brillantes
        self.hsv_upper = np.array([180, 100, 255])  # Valores HSV máximos para objetos brillantes

        # Parámetros para objetos metálicos más oscuros
        self.dark_metal_lower = np.array([0, 0, 20])
        self.dark_metal_upper = np.array([180, 50, 80])

        # Parámetros para contornos
        self.min_contour_area = 500  # Área mínima para considerar un contorno
        self.max_contour_area = 50000  # Área máxima para considerar un contorno

        # Historial para suavizar detecciones
        self.history = []
        self.history_size = 5

        # Umbral de conductividad para clasificación de metales
        self.conductivity_threshold = 0.65

    def detect_traditional(self, frame):
        """
        Implementación original del método detect.
        Este es tu código actual sin cambios.
        """
        # Convertir a HSV para mejor detección de brillos y colores metálicos
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear máscaras para detección
        mask_bright = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask_dark = cv2.inRange(hsv, self.dark_metal_lower, self.dark_metal_upper)

        # Combinar máscaras
        mask = cv2.bitwise_or(mask_bright, mask_dark)

        # Mejora de la máscara con operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filtrar por área
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # Obtener centro del contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Obtener rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)

            # Estimar conductividad basada en brillo y textura
            roi = frame[y:y + h, x:x + w]

            # Verificar que el ROI no esté vacío
            if roi.size == 0:
                continue

            # Convertir a escala de grises
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Calcular el brillo medio (indicador simple de reflectividad)
            avg_brightness = np.mean(roi_gray)

            # Calcular varianza de textura (indicador de rugosidad)
            texture_variance = np.var(roi_gray)

            # Estimar conductividad (simple heurística: objetos más brillantes y uniformes)
            estimated_conductivity = min(1.0, max(0.0,
                                                (avg_brightness / 255.0) * 0.7 +
                                                (1.0 - min(1.0, texture_variance / 1000.0)) * 0.3))

            # Clasificar el tipo de objeto
            obj_type = 'metallic' if estimated_conductivity > 0.6 else 'conductive'

            # Crear objeto detectado
            detected_object = {
                'contour': contour,
                'center': (cx, cy),
                'area': area,
                'bounding_box': (x, y, w, h),
                'type': obj_type,
                'estimated_conductivity': estimated_conductivity
            }

            detected_objects.append(detected_object)

        return detected_objects

    def detect(self, frame):
        """
        Versión mejorada que usa un sistema de puntuación ponderada.
        """
        # 1. Usar el método tradicional para detección inicial
        potential_objects = self.detect_traditional(frame)

        # 2. Para cada objeto potencial, calcular múltiples características
        final_objects = []
        for obj in potential_objects:
            x, y, w, h = obj['bounding_box']
            roi = frame[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            # Características de brillo y textura
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(roi_gray)
            texture_variance = np.var(roi_gray)

            # Nuevas características
            # a. Uniformidad de color (los metales tienden a tener colores más uniformes)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(roi_hsv)
            hue_std = np.std(h_channel)
            sat_std = np.std(s_channel)
            color_uniformity = 1.0 - min(1.0, (hue_std/50.0 + sat_std/50.0)/2)

            # b. Reflectividad (basada en valor HSV y varianza)
            v_mean = np.mean(v_channel)
            v_std = np.std(v_channel)
            reflectivity = (v_mean/255.0) * (1.0 - min(1.0, v_std/50.0))

            # c. Bordes definidos (metales suelen tener bordes nítidos)
            edges = cv2.Canny(roi_gray, 100, 200)
            edge_density = np.sum(edges) / (roi.shape[0] * roi.shape[1])

            # Sistema de puntuación ponderada
            scores = {
                'brightness': avg_brightness / 255.0,
                'texture_smoothness': 1.0 - min(1.0, texture_variance / 2000.0),
                'color_uniformity': color_uniformity,
                'reflectivity': reflectivity,
                'edge_definition': min(1.0, edge_density * 10)
            }

            # Pesos para cada característica
            weights = {
                'brightness': 0.2,
                'texture_smoothness': 0.25,
                'color_uniformity': 0.3,
                'reflectivity': 0.15,
                'edge_definition': 0.1
            }

            # Calcular puntuación final
            final_score = sum(scores[k] * weights[k] for k in weights)

            # Clasificar basado en puntuación
            obj['estimated_conductivity'] = final_score
            obj['type'] = 'metallic' if final_score > self.conductivity_threshold else 'conductive'

            # Guardar puntuaciones individuales para depuración
            obj['feature_scores'] = scores

            final_objects.append(obj)

        # Mantener el historial actualizado
        self.history.append(final_objects)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return final_objects

    def adjust_detection_parameters(self, sensitivity=None, min_area=None, max_area=None, conductivity_threshold=None):
        """
        Ajusta los parámetros de detección.

        Args:
            sensitivity: Valor entre 0 y 1 para ajustar la sensibilidad global
            min_area: Área mínima para considerar un contorno
            max_area: Área máxima para considerar un contorno
            conductivity_threshold: Umbral para clasificar objetos como metálicos
        """
        if sensitivity is not None:
            # Ajustar umbrales HSV basados en sensibilidad
            sensitivity = max(0.0, min(1.0, sensitivity))

            # Ajustar valores inferiores
            self.hsv_lower[1] = int(50 * (1 - sensitivity))  # Menos saturación para más sensibilidad
            self.hsv_lower[2] = int(150 - 50 * sensitivity)  # Menos brillo para más sensibilidad

            # Ajustar valores para metales oscuros
            self.dark_metal_lower[2] = int(20 * (1 - sensitivity))
            self.dark_metal_upper[2] = int(80 + 20 * sensitivity)

        # Ajustar áreas de contorno
        if min_area is not None:
            self.min_contour_area = max(100, min_area)

        if max_area is not None:
            self.max_contour_area = max(self.min_contour_area * 2, max_area)

        # Ajustar umbral de conductividad
        if conductivity_threshold is not None:
            self.conductivity_threshold = max(0.3, min(0.9, conductivity_threshold))

    def draw_debug_info(self, frame, detected_objects):
        """
        Dibuja información de depuración sobre las características detectadas.
        """
        debug_frame = frame.copy()

        for obj in detected_objects:
            if 'feature_scores' not in obj:
                continue

            # Obtener puntuaciones
            scores = obj['feature_scores']
            x, y, w, h = obj['bounding_box']

            # Dibujar rectángulo alrededor del objeto
            color = (0, 255, 0) if obj['type'] == 'metallic' else (0, 165, 255)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)

            # Mostrar puntuación final
            cv2.putText(debug_frame, f"{obj['estimated_conductivity']:.2f}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mostrar puntuaciones individuales (desplazadas)
            y_offset = y + h + 15
            for key, score in scores.items():
                text = f"{key}: {score:.2f}"
                cv2.putText(debug_frame, text, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15

        return debug_frame
