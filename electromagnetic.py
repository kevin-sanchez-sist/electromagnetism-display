"""
Simulador de campos electromagnéticos basado en la Ley de Faraday
"""

import numpy as np
import time
import math


class ElectromagneticField:
    def __init__(self):
        # Parámetros del campo electromagnético
        self.field_strength = 1.0  # Tesla
        self.field_frequency = 1.0  # Hz
        self.magnet_position = (0.5, 0.5)  # Posición normalizada (x, y) entre 0 y 1

        # Parámetros para la Ley de Faraday
        self.coil_turns = 10  # Número de vueltas en la bobina (asumido para objetos)
        self.permeability = 1.0  # Permeabilidad magnética relativa (se ajustará por objeto)

        # Para seguimiento del tiempo y animación
        self.start_time = time.time()

    def set_field_strength(self, strength):
        """Establece la intensidad del campo magnético en Tesla"""
        self.field_strength = strength

    def set_field_frequency(self, frequency):
        """Establece la frecuencia de oscilación del campo en Hz"""
        self.field_frequency = frequency

    def set_magnet_position(self, position):
        """Establece la posición normalizada del imán (x, y)"""
        self.magnet_position = position

    def set_coil_turns(self, turns):
        """Establece el número de vueltas en la bobina"""
        self.coil_turns = turns

    def generate_field(self, width, height):
        """
        Genera los datos del campo electromagnético para visualización.

        Args:
            width: Ancho de la imagen en píxeles
            height: Alto de la imagen en píxeles

        Returns:
            Diccionario con los datos del campo:
            {
                'field_vectors': array de vectores (dx, dy, intensidad),
                'field_scalar': array 2D con la intensidad del campo,
                'flux_lines': lista de líneas de flujo [(x1,y1), (x2,y2), ...],
                'current_phase': fase actual del campo (0-2π)
            }
        """
        # Calcular la fase actual basada en el tiempo y la frecuencia
        current_time = time.time() - self.start_time
        current_phase = (current_time * self.field_frequency) % 1.0 * 2 * np.pi

        # Crear mallas para coordenadas x e y
        grid_size = 20  # Número de puntos en la cuadrícula para visualización
        x = np.linspace(0, width, grid_size)
        y = np.linspace(0, height, grid_size)
        X, Y = np.meshgrid(x, y)

        # Convertir posición del imán a coordenadas de píxeles
        magnet_x = self.magnet_position[0] * width
        magnet_y = self.magnet_position[1] * height

        # Calcular distancias desde cada punto de la cuadrícula al imán
        distances = np.sqrt((X - magnet_x) ** 2 + (Y - magnet_y) ** 2)
        # Evitar división por cero
        distances = np.maximum(distances, 1.0)

        # Calcular intensidad del campo (varía con 1/r²)
        field_intensity = self.field_strength * (100.0 / distances ** 2)
        field_intensity = np.minimum(field_intensity, self.field_strength * 10)  # Limitar máximo

        # Calcular vectores de dirección del campo (desde el imán hacia afuera)
        dx = (X - magnet_x) / distances
        dy = (Y - magnet_y) / distances

        # Modular la intensidad con la fase actual para crear oscilación
        oscillating_factor = 0.5 * np.sin(current_phase) + 0.5
        field_intensity = field_intensity * oscillating_factor

        # Generar líneas de flujo magnético (para visualización)
        flux_lines = self._generate_flux_lines(magnet_x, magnet_y, width, height, current_phase)

        # Crear array de campo escalar para toda la imagen
        scalar_field = np.zeros((height, width))

        # Interpolar el campo para toda la imagen
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Coordenadas de la celda actual
                x0, y0 = int(x[j]), int(y[i])
                x1, y1 = int(x[j + 1]), int(y[i + 1])

                # Asegurarse de que las coordenadas están dentro de los límites
                if x1 >= width: x1 = width - 1
                if y1 >= height: y1 = height - 1

                # Valor de intensidad en esta celda
                val = field_intensity[i, j]

                # Llenar el área de la celda en el campo escalar
                scalar_field[y0:y1, x0:x1] = val

        return {
            'field_vectors': (dx, dy, field_intensity),
            'field_scalar': scalar_field,
            'flux_lines': flux_lines,
            'current_phase': current_phase
        }

    def _generate_flux_lines(self, magnet_x, magnet_y, width, height, phase):
        """
        Genera líneas de flujo magnético alrededor del imán.
        """
        flux_lines = []
        num_lines = 16  # Número de líneas de flujo

        for i in range(num_lines):
            angle = i * (2 * np.pi / num_lines)

            # Crear línea desde el imán hacia afuera
            line = []

            # Factor oscilante para longitud de líneas
            oscillation = 0.7 + 0.3 * np.sin(phase + i * np.pi / 8)

            # Longitud base de la línea
            line_length = min(width, height) * 0.4 * oscillation

            # Número de puntos en la línea
            num_points = 20

            for t in range(num_points):
                # Parámetro de distancia normalizado
                t_norm = t / (num_points - 1)

                # Distancia desde el imán (aumenta de forma no lineal)
                distance = line_length * (t_norm ** 0.5)

                # Calcular posición
                x = magnet_x + distance * np.cos(angle)
                y = magnet_y + distance * np.sin(angle)

                # Añadir ondulación a las líneas para simular dinamismo
                wave_factor = 5.0 * np.sin(phase + 10 * t_norm)
                x += wave_factor * np.sin(angle + np.pi / 2)
                y -= wave_factor * np.cos(angle + np.pi / 2)

                # Verificar límites
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                line.append((x, y))

            flux_lines.append(line)

        return flux_lines

    def update_field_for_objects(self, field_data, detected_objects):
        """
        Actualiza el campo electromagnético basado en objetos detectados.

        Args:
            field_data: Datos del campo generados por generate_field()
            detected_objects: Lista de objetos detectados

        Returns:
            Datos de campo actualizados
        """
        # Si no hay objetos, devolver el campo sin cambios
        if not detected_objects:
            return field_data

        # Obtener datos del campo
        dx, dy, intensity = field_data['field_vectors']
        scalar_field = field_data['field_scalar']

        # Para cada objeto, modificar el campo
        for obj in detected_objects:
            # Obtener propiedades del objeto
            center = obj['center']
            conductivity = obj['estimated_conductivity']
            area = obj['area']

            # Ajustar la permeabilidad basada en conductividad
            permeability = 1.0 + conductivity * 10.0  # Mayor conductividad = mayor permeabilidad

            # Radio de influencia proporcional al área del objeto
            influence_radius = np.sqrt(area / np.pi) * 3

            # Modificar el campo escalar basado en permeabilidad
            height, width = scalar_field.shape
            for y in range(height):
                for x in range(width):
                    # Distancia desde este punto al centro del objeto
                    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                    # Si está dentro del radio de influencia
                    if distance <= influence_radius:
                        # Factor de modificación basado en distancia
                        mod_factor = 1.0 - (distance / influence_radius)

                        # Aumentar campo dentro del objeto conductivo
                        scalar_field[y, x] *= (1.0 + mod_factor * (permeability - 1.0))

        # Actualizar el campo vectorial para mantener consistencia
        field_data['field_scalar'] = scalar_field

        return field_data

    def calculate_induced_voltage(self, detected_objects):
        """
        Calcula la fuerza electromotriz inducida en objetos conductores según la Ley de Faraday.

        Args:
            detected_objects: Lista de objetos detectados

        Returns:
            Lista de voltajes inducidos para cada objeto
        """
        induced_voltages = []

        # Si no hay objetos, devolver lista vacía
        if not detected_objects:
            return induced_voltages

        # Para cada objeto, calcular voltaje inducido
        for obj in detected_objects:
            # Obtener propiedades del objeto
            conductivity = obj['estimated_conductivity']
            area = obj['area']

            # Estimar área efectiva en metros cuadrados (aproximación)
            effective_area = area / 1000000  # Convertir de píxeles² a metros²

            # Estimar número de vueltas basado en tipo de objeto
            # (asumiendo que objetos metálicos tienen comportamiento similar a una espira única)
            effective_turns = self.coil_turns

            # Calcular flujo magnético
            magnetic_flux = self.field_strength * effective_area

            # Calcular tasa de cambio del flujo magnético
            # Usando la derivada de B*A*cos(ωt) con respecto a t
            flux_change_rate = magnetic_flux * self.field_frequency * 2 * np.pi

            # Aplicar la Ley de Faraday: ε = -N * dΦ/dt
            # El signo negativo es incluido en la física, pero para visualización usamos valor absoluto
            induced_voltage = effective_turns * flux_change_rate * conductivity

            # Agregar oscilación basada en la fase actual
            current_time = time.time() - self.start_time
            current_phase = (current_time * self.field_frequency) % 1.0 * 2 * np.pi
            oscillation_factor = abs(np.sin(current_phase))

            # Voltaje final con oscilación
            final_voltage = induced_voltage * oscillation_factor

            induced_voltages.append({
                'object': obj,
                'voltage': final_voltage,
                'oscillation_phase': current_phase
            })

        return induced_voltages