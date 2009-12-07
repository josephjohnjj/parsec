#include <dplasma.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

extern int yyparse();
extern int dplasma_lineno;
extern FILE *yyin;
char *yyfilename;

int main(int argc, char *argv[])
{
    int d;
    
    if( argc != 1 ) {
        yyin = fopen(argv[1], "r");
        if( yyin == NULL ) {
            fprintf(stderr, "unable to open input file %s: %s\n", argv[1], strerror(errno));
            exit(1);
        }
        yyfilename = strdup(argv[1]);
    } else {
        yyfilename = strdup("(stdin)");
    }

    dplasma_lineno = 1;
	if( (d = yyparse()) > 0 ) {
        exit(1);
    }

    dplasma_dump_all_c(stdout);

	return 0;
}
