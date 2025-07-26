"""
Módulo de visualización para el Visualizador de Campos Electromagnéticos
"""

import cv2
import numpy as np
import math

class FieldVisualizer:
    def __init__(self):
        # Opciones de visualización
        self.field_color = (0, 255, 255)  # Rojo para el campo (en BGR)
        self.field_opacity = 0.4  # Opacidad del campo
        self.arrow_size = 15  # Tamaño de las flechas de campo
        self.flux_line_color = (255, 160, 0)  # Azul claro para líneas de flujo (en BGR)
        self.flux_line_thickness = 1  # Grosor de las líneas de flujo

        # Generar mapa de colores para visualización del campo
        self.color_map = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8),
            cv2.COLORMAP_JET
        )

        # Fuente para texto
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_color = (255, 255, 255)  # Blanco
        self.font_thickness = 1

    def overlay_field(self, frame, field_data):
        """
        Superpone la visualización del campo electromagnético sobre el frame.

        Args:
            frame: Imagen BGR de la cámara
            field_data: Datos del campo generados por ElectromagneticField

        Returns:
            Frame con el campo superpuesto
        """
        # Crear copia del frame
        result = frame.copy()

        # Obtener dimensiones
        height, width, _ = frame.shape

        # Visualizar campo escalar
        self._visualize_scalar_field(result, field_data['field_scalar'])

        # Visualizar vectores de campo
        self._visualize_vector_field(result, field_data['field_vectors'], width, height)

        # Visualizar líneas de flujo
        self._visualize_flux_lines(result, field_data['flux_lines'], field_data['current_phase'])

        return result

    def _visualize_scalar_field(self, frame, scalar_field):
        """
        Visualiza el campo escalar como un mapa de calor semitransparente.
        """
        # Normalizar a rango 0-255
        normalized_field = cv2.normalize(scalar_field, None, 0, 255, cv2.NORM_MINMAX)
        normalized_field = normalized_field.astype(np.uint8)

        # Aplicar mapa de colores
        colored_field = cv2.applyColorMap(normalized_field, cv2.COLORMAP_JET)

        # Crear máscara de opacidad basada en intensidad
        opacity = np.zeros_like(normalized_field)
        opacity = np.minimum(normalized_field * 2, 255).astype(np.uint8)

        # Aplicar máscara alpha
        mask = opacity > 20  # Solo mostrar donde la intensidad es significativa

        # Superponer el campo sobre el frame original
        for c in range(3):  # Para cada canal de color
            frame[:, :, c] = np.where(
                mask,
                frame[:, :, c] * (1 - self.field_opacity) + colored_field[:, :, c] * self.field_opacity,
                frame[:, :, c]
            )

    def _visualize_vector_field(self, frame, field_vectors, width, height):
        """
        Visualiza el campo vectorial como flechas.
        """
        dx, dy, intensity = field_vectors

        # Dibujar flechas en una cuadrícula más espaciada para claridad
        grid_size = 20
        step_x = width // grid_size
        step_y = height // grid_size

        for i in range(0, grid_size):
            for j in range(0, grid_size):
                # Índices de la cuadrícula
                grid_i, grid_j = i, j

                # Coordenadas en píxeles
                x = j * step_x
                y = i * step_y

                # Verificar límites
                if grid_i >= dx.shape[0] or grid_j >= dx.shape[1]:
                    continue

                # Obtener vector e intensidad
                vector_dx = dx[grid_i, grid_j]
                vector_dy = dy[grid_i, grid_j]
                vector_intensity = intensity[grid_i, grid_j]

                # Normalizar vector
                magnitude = np.sqrt(vector_dx**2 + vector_dy**2)
                if magnitude > 0:
                    vector_dx /= magnitude
                    vector_dy /= magnitude

                # Escalar el tamaño de la flecha según la intensidad
                arrow_length = int(self.arrow_size * min(1.0, vector_intensity))

                # Calcular punto final de la flecha
                end_x = int(x + arrow_length * vector_dx)
                end_y = int(y + arrow_length * vector_dy)

                # Color basado en intensidad
                intensity_normalized = min(1.0, vector_intensity / 5.0) * 255
                intensity_color = self.color_map[int(intensity_normalized)][0].tolist()

                # Dibujar flecha
                cv2.arrowedLine(
                    frame,
                    (x, y),
                    (end_x, end_y),
                    intensity_color,
                    1,  # Grosor
                    cv2.LINE_AA,  # Tipo de línea
                    0,  # Desplazamiento
                    0.3  # Tamaño de la punta de flecha
                )

    def _visualize_flux_lines(self, frame, flux_lines, phase):
        """
        Visualiza las líneas de flujo magnético.
        """
        for line in flux_lines:
            # Convertir a array de NumPy para dibujo
            points = np.array(line, dtype=np.int32)

            # Modular color por fase para efecto pulsante
            phase_factor = 0.7 + 0.3 * np.sin(phase)
            color = (
                int(self.flux_line_color[0] * phase_factor),
                int(self.flux_line_color[1] * phase_factor),
                int(self.flux_line_color[2] * phase_factor)
            )

            # Dibujar línea
            for i in range(len(points) - 1):
                cv2.line(
                    frame,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    color,
                    self.flux_line_thickness,
                    cv2.LINE_AA
                )

    def draw_objects(self, frame, detected_objects):
        """
        Dibuja contornos y etiquetas para los objetos detectados.

        Args:
            frame: Imagen BGR de la cámara
            detected_objects: Lista de objetos detectados

        Returns:
            Frame con objetos marcados
        """
        result = frame.copy()

        for obj in detected_objects:
            contour = obj['contour']
            center = obj['center']
            bbox = obj['bounding_box']
            obj_type = obj['type']
            conductivity = obj['estimated_conductivity']

            # Determinar color basado en el tipo de objeto
            if obj_type == 'metallic':
                color = (0, 255, 255)  # Amarillo
            else:
                color = (0, 165, 255)  # Naranja

            # Dibujar contorno
            cv2.drawContours(result, [contour], 0, color, 2)

            # Dibujar centro
            cv2.circle(result, center, 3, (0, 0, 255), -1)

            # Dibujar caja delimitadora
            x, y, w, h = bbox
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)

            # Agregar etiqueta
            label = f"{obj_type.capitalize()}: {conductivity:.2f}"
            text_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(result, (x, y-text_size[1]-5), (x+text_size[0], y), color, -1)
            cv2.putText(result, label, (x, y-5), self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        return result

    def show_induced_voltage(self, frame, induced_voltages):
        """
        Muestra el voltaje inducido para cada objeto.

        Args:
            frame: Imagen BGR de la cámara
            induced_voltages: Lista de voltajes inducidos para cada objeto

        Returns:
            Frame con información de voltaje
        """
        result = frame.copy()

        for voltage_data in induced_voltages:
            obj = voltage_data['object']
            voltage = voltage_data['voltage']
            phase = voltage_data['oscillation_phase']

            # Obtener centro y área
            center = obj['center']
            bbox = obj['bounding_box']

            # Formatear texto de voltaje
            voltage_text = f"FEM = {voltage*1000:.2f} mV"

            # Calcular posición para el texto (encima del objeto)
            x, y = center[0], bbox[1] - 20

            # Factor pulsante basado en fase
            pulse_factor = 0.5 + 0.5 * np.sin(phase)

            # Color basado en magnitud del voltaje y fase
            color_intensity = min(255, int(voltage * 10000))
            color = (
                0,
                int(255 * pulse_factor),
                color_intensity
            )

            # Dibujar fondo para el texto
            text_size, _ = cv2.getTextSize(voltage_text, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(
                result,
                (x - text_size[0]//2 - 5, y - text_size[1] - 5),
                (x + text_size[0]//2 + 5, y + 5),
                (0, 0, 0),
                -1
            )

            # Dibujar texto
            cv2.putText(
                result,
                voltage_text,
                (x - text_size[0]//2, y),
                self.font,
                self.font_scale,
                color,
                self.font_thickness,
                cv2.LINE_AA
            )

            # Dibujar símbolo de voltaje inducido (círculos concéntricos)
            radius = int(10 + 5 * pulse_factor)
            cv2.circle(result, center, radius, color, 2)
            cv2.circle(result, center, radius//2, color, 1)

            # Dibujar dirección del flujo de corriente inducida
            angle_step = 2 * np.pi / 8
            for i in range(8):
                angle = i * angle_step + phase
                x1 = int(center[0] + (radius + 5) * np.cos(angle))
                y1 = int(center[1] + (radius + 5) * np.sin(angle))
                x2 = int(center[0] + (radius + 15) * np.cos(angle))
                y2 = int(center[1] + (radius + 15) * np.sin(angle))

                # Solo dibujar flechas alternas para indicar dirección
                if i % 2 == 0:
                    cv2.arrowedLine(result, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, 0, 0.3)

        return result

    def overlay_faraday_equation(self, frame):
        """
        Superpone la ecuación de la Ley de Faraday en la esquina superior izquierda.
        """
        result = frame.copy()

        equation = "Ley de Faraday: FEM = -N*d(phi)B/dt"

        # Dibujar fondo negro semitransparente
        text_size, _ = cv2.getTextSize(equation, self.font, 0.7, 2)
        cv2.rectangle(
            result,
            (10, 10),
            (10 + text_size[0] + 10, 10 + text_size[1] + 10),
            (0, 0, 0),
            -1
        )

        # Dibujar ecuación
        cv2.putText(
            result,
            equation,
            (15, 10 + text_size[1]),
            self.font,
            0.7,  # Escala de fuente más grande
            (255, 255, 255),  # Blanco
            2,  # Grosor mayor
            cv2.LINE_AA
        )

        return result