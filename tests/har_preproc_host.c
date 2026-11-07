/* Хост-обгортка для перевірки har_preproc.c на ПК (не для прошивки).
 *
 * Читає зі stdin рядки з HAR_CHANNELS чисел — один відлік на рядок.
 * Режим задається аргументом, бо стан фільтра спільний для обох шляхів,
 * тому кожен перевіряється окремим запуском:
 *
 *   filter   друкує "F <канали...>" — вихід har_preproc_filter()
 *   window   друкує "W <k> <208*6 чисел>" — чергове вікно з har_preproc_push()
 *            і "N <k> <208*6 чисел>" — те саме після har_preproc_normalize()
 *   gap <m>  те саме, що window, але перед останнім відліком кожного рядка
 *            повідомляється <m> втрачених відліків (har_preproc_push_missed);
 *            друкує ще "C <k> <номер відліку>" — на якому відліку вийшло вікно
 *
 * Збирається і запускається з tests/test_har_preproc.py.
 */
#include <stdio.h>
#include <string.h>

#include "har_preproc.h"

static int read_sample(float sample[HAR_CHANNELS])
{
	for (int c = 0; c < HAR_CHANNELS; c++) {
		if (scanf("%f", &sample[c]) != 1) {
			return 0;
		}
	}
	return 1;
}

static void print_row(const char *tag, int index, const float *values, int count)
{
	if (index >= 0) {
		printf("%s %d", tag, index);
	} else {
		printf("%s", tag);
	}
	for (int i = 0; i < count; i++) {
		printf(" %.9g", (double) values[i]);
	}
	printf("\n");
}

int main(int argc, char **argv)
{
	const char *mode = (argc > 1) ? argv[1] : "filter";
	float sample[HAR_CHANNELS];
	float filtered[HAR_CHANNELS];
	float copy[HAR_WINDOW_SAMPLES * HAR_CHANNELS];
	int windows = 0;

	har_preproc_init();

	if (strcmp(mode, "filter") == 0) {
		while (read_sample(sample)) {
			har_preproc_filter(sample, filtered);
			print_row("F", -1, filtered, HAR_CHANNELS);
		}
		return 0;
	}

	/* У режимі gap кожен відлік подається як такий, перед яким загубилося
	   `missed` відліків — так перевіряється добивання і розрив вікна. */
	unsigned long missed = 0;
	if (strcmp(mode, "gap") == 0) {
		if (argc < 3 || sscanf(argv[2], "%lu", &missed) != 1) {
			fprintf(stderr, "usage: har_preproc_host gap <missed>\n");
			return 2;
		}
	} else if (strcmp(mode, "window") != 0) {
		fprintf(stderr, "usage: har_preproc_host [filter|window|gap <missed>]\n");
		return 2;
	}

	long index = 0;
	while (read_sample(sample)) {
		index++;
		const int ready = missed ? har_preproc_push_missed(sample, (uint32_t) missed)
		                         : har_preproc_push(sample);
		if (!ready) {
			continue;
		}
		printf("C %d %ld\n", windows, index);
		const float *w = har_preproc_window();
		const int n = HAR_WINDOW_SAMPLES * HAR_CHANNELS;

		print_row("W", windows, w, n);
		memcpy(copy, w, sizeof(copy));
		har_preproc_normalize(copy);
		print_row("N", windows, copy, n);
		windows++;
	}
	return 0;
}
